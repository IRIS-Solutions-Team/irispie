"""
"""


#[

from __future__ import annotations

import numpy as _np
import scipy as _sp
import itertools as _it
import documark as _dm

from ..incidences import main as _incidences
from ..fords import covariances as _covariances
from .. import quantities as _quantities
from .. import namings as _namings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Real

#]




class Inlay:
    r"""
    ................................................................................
    ==Inlay: Core Class for Managing Covariance Calculations==

    The `Inlay` class provides advanced methods for managing and calculating 
    covariances, autocovariances, and standard deviations of shocks in dynamic 
    models. It leverages efficient numerical computations to handle variants, 
    scaling, and transformations.

    Attributes:
        _variants: Stores all model variants for calculations.
        _invariant: Provides access to invariant quantities for model setup.
    ................................................................................
    """

    @_dm.reference(category="parameters", )
    def rescale_stds(
        self,
        factor: Real,
        *,
        kind: _quantities.QuantityKind | None = None,
    ) -> None:
        """
................................................................................

==Rescale the standard deviations of model shocks==

Adjust the standard deviations of the model shocks by a specified factor. 
This method allows scaling the standard deviations for the shocks in the 
model based on the provided factor.

    self.rescale_stds(
        factor,
        kind=None,
    )


### Input arguments ###


???+ input "factor"
    A real non-negative number by which to scale the standard deviations of the
    model shocks. This value is used to multiply the existing standard
    deviations.

???+ input "kind"
    An optional parameter to narrow down the types of shocks to rescale. It
    can be one, or a combination, of the following:

    * `ir.UNANTICIPATED_STD`
    * `ir.ANTICIPATED_STD`
    * `ir.MEASUREMENT_STD`

    If `None`, the standard deviations of all shocks will be rescaled.


### Returns ###


???+ returns "None"
    This method does not return any value but modifies the standard deviations 
    of model shocks in-place, rescaling them.

................................................................................
        """
        if 1.0 * factor <= 0:
            raise ValueError("The scaling factor must be a real non-negative number")
        std_qids = self._get_std_qids(kind=kind, )
        for v in self._variants:
            v.rescale_values("levels", factor, std_qids, )

    def get_acov(
        self,
        /,
        up_to_order: int = 0,
        return_dimension_names: bool = True,
        unpack_singleton: bool = True,
    ) -> tuple[list[_np.ndarray] | _np.ndarray, _namings.DimensionNames] | list[_np.ndarray] | _np.ndarray:
        r"""
        ................................................................................
        ==Compute Autocovariance of Variables==

        Computes the asymptotic autocovariance of model variables up to a specified
        order.

        This method computes autocovariance matrices for transition and measurement
        variables, optionally returning dimension names.

        ### Input arguments ###
        ???+ input "up_to_order"
            The maximum order of autocovariance to compute. Defaults to 0.
        ???+ input "return_dimension_names"
            Whether to return the names of dimensions in the result. Defaults to `True`.
        ???+ input "unpack_singleton"
            Whether to unpack singleton results into simpler structures. Defaults to `True`.

        ### Returns ###
        ???+ returns "tuple"
            A tuple containing:
            - Autocovariance matrices as arrays or lists of arrays.
            - Dimension names if `return_dimension_names` is `True`.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            acov, dim_names = obj.get_acov(up_to_order=1)
            print(acov, dim_names)
        ```
        ................................................................................
        """
        #
        # Combine vectors of transition and measurement tokens, and select
        # those with zero shift only using a boolex
        system_vector, boolex_zero_shift = _get_system_vector(self, )
        #
        # Tuple element cov_by_variant[variant_index][order] is an N-by-N
        # covariance matrix
        cov_by_variant = [
            self._getv_autocov(v, boolex_zero_shift, up_to_order=up_to_order, )
            for v in self._variants
        ]
        #
        # Tuple element cov_by_order[order] is an N-by-N covariance matrix
        # (for singleton models) or an N-by-N-by-num_variants covariance
        # matrix (for nonsingleton models)
        # stack_func = _STACK_FUNC_FACTORY[self.is_singleton]
        # cov_by_order = tuple(
        #     stack_func(tuple(cov[order] for cov in cov_by_variant))
        #     for order in range(up_to_order + 1)
        # )
        #
        cov_by_variant = self.unpack_singleton(
            cov_by_variant,
            unpack_singleton=unpack_singleton,
        )
        #
        if return_dimension_names:
            dimension_names = _get_dimension_names(self, system_vector, )
            return cov_by_variant, dimension_names
        else:
            return cov_by_variant

    def get_acorr(
        self,
        /,
        acov: tuple[_np.ndarray, ..., tuple[str], tuple[str]] | None = None,
        up_to_order: int = 0,
        unpack_singleton: bool = True,
        return_dimension_names: bool = True,
    ) -> tuple[list[_np.ndarray] | _np.ndarray, _namings.DimensionNames] | list[_np.ndarray] | _np.ndarray:
        r"""
        ................................................................................
        ==Compute Autocorrelation of Variables==

        Computes the asymptotic autocorrelation of model variables up to a specified
        order.

        ### Input arguments ###
        ???+ input "acov"
            Precomputed autocovariance values. If `None`, autocovariances are computed internally.
        ???+ input "up_to_order"
            The maximum order of autocorrelation to compute. Defaults to 0.
        ???+ input "unpack_singleton"
            Whether to unpack singleton results into simpler structures. Defaults to `True`.
        ???+ input "return_dimension_names"
            Whether to return the names of dimensions in the result. Defaults to `True`.

        ### Returns ###
        ???+ returns "tuple"
            A tuple containing:
            - Autocorrelation matrices as arrays or lists of arrays.
            - Dimension names if `return_dimension_names` is `True`.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            acorr, dim_names = obj.get_acorr(up_to_order=1)
            print(acorr, dim_names)
        ```
        ................................................................................
        """
        acov_by_variant = acov
        if acov_by_variant is None:
            acov_by_variant, *_ = self.get_acov(
                up_to_order=up_to_order,
                return_dimension_names=False,
                unpack_singleton=False,
            )
        acov_by_variant = self.repack_singleton(acov_by_variant, )
        acorr_by_variant = [
            _covariances.acorr_from_acov(i)
            for i in acov_by_variant
        ]
        acorr_by_variant = self.unpack_singleton(
            acorr_by_variant,
            unpack_singleton=unpack_singleton,
        )
        if return_dimension_names:
            system_vector, *_ = _get_system_vector(self, )
            dimension_names = _get_dimension_names(self, system_vector, )
            return acorr_by_variant, dimension_names,
        else:
            return acorr_by_variant

    def _getv_autocov(
        self,
        variant: _variants.Variant,
        boolex_zero_shift: tuple[bool, ...] | Ellipsis,
        /,
        up_to_order: int = 0,
    ) -> _np.ndarray:
        r"""
        ................................................................................
        ==Compute Autocovariance for a Variant==

        Calculates the autocovariance of variables for a specific variant up to
        a given order. This internal method filters and selects elements of the 
        covariance matrix based on the shift value.

        ### Input arguments ###
        ???+ input "variant"
            A specific variant of the model for which autocovariance is computed.
        ???+ input "boolex_zero_shift"
            A tuple indicating whether variables have a zero shift.
        ???+ input "up_to_order"
            The maximum order of autocovariance to compute. Defaults to 0.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A NumPy array containing the computed autocovariance matrices.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            autocov = obj._getv_autocov(variant, boolex_zero_shift, up_to_order=2)
            print(autocov)
        ```
        ................................................................................
        """
        def select(cov: _np.ndarray, /, ) -> _np.ndarray:
            """
            Select elements of solution vectors with zero shift only
            """
            cov = cov[boolex_zero_shift, :]
            cov = cov[:, boolex_zero_shift]
            return cov
        cov_u = self._getv_cov_u(variant, )
        cov_w = self._getv_cov_w(variant, )
        cov_by_order = _covariances.get_autocov_square(variant.solution, cov_u, cov_w, up_to_order, )
        return tuple(select(cov, ) for cov in cov_by_order)

    def get_stdvec_unanticipated_shocks(self, /, ):
        r"""
        ................................................................................
        ==Retrieve Standard Deviations of Unanticipated Shocks==

        Extracts the standard deviations for unanticipated shocks from the model
        variants. This method operates on each variant individually.

        ### Input arguments ###
        ???+ input "None"
            This method does not take any input arguments.

        ### Returns ###
        ???+ returns "list"
            A list containing arrays of standard deviations for unanticipated shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            std_vec = obj.get_stdvec_unanticipated_shocks()
            print(std_vec)
        ```
        ................................................................................
        """
        std_u = [
            self._getv_std_u(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(std_u, )

    def get_stdvec_measurement_shocks(self, /, ):
        r"""
        ................................................................................
        ==Retrieve Standard Deviations of Measurement Shocks==

        Extracts the standard deviations for measurement shocks from the model
        variants. This method operates on each variant individually.

        ### Input arguments ###
        ???+ input "None"
            This method does not take any input arguments.

        ### Returns ###
        ???+ returns "list"
            A list containing arrays of standard deviations for measurement shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            std_vec = obj.get_stdvec_measurement_shocks()
            print(std_vec)
        ```
        ................................................................................
        """
        std_w = [
            self._getv_std_w(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(std_w, )

    def get_cov_unanticipated_shocks(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        r"""
        ................................................................................
        ==Retrieve Covariance Matrices of Unanticipated Shocks==

        Computes the covariance matrices of unanticipated shocks across all model
        variants.

        ### Input arguments ###
        ???+ input "unpack_singleton"
            Whether to unpack singleton results into simpler structures. Defaults to `True`.

        ### Returns ###
        ???+ returns "list"
            A list of covariance matrices for unanticipated shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            cov_u = obj.get_cov_unanticipated_shocks()
            print(cov_u)
        ```
        ................................................................................
        """
        cov_u = [
            self._getv_cov_u(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(cov_u, unpack_singleton=unpack_singleton, )

    def get_cov_measurement_shocks(
        self,
        /,
        unpack_singleton: bool = True,
    ):
        r"""
        ................................................................................
        ==Retrieve Covariance Matrices of Measurement Shocks==

        Computes the covariance matrices of measurement shocks across all model
        variants.

        ### Input arguments ###
        ???+ input "unpack_singleton"
            Whether to unpack singleton results into simpler structures. Defaults to `True`.

        ### Returns ###
        ???+ returns "list"
            A list of covariance matrices for measurement shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            cov_w = obj.get_cov_measurement_shocks()
            print(cov_w)
        ```
        ................................................................................
        """
        cov_w = [
            self._getv_cov_w(v, )
            for v in self._variants
        ]
        return self.unpack_singleton(cov_w, unpack_singleton=unpack_singleton, )

    def _getv_std_u(self, variant, /, ):
        r"""
        ................................................................................
        ==Retrieve Standard Deviations of Unanticipated Shocks for a Variant==

        Extracts the standard deviations of unanticipated shocks for a specific
        model variant. This method retrieves values from the associated solution 
        vectors.

        ### Input arguments ###
        ???+ input "variant"
            The model variant for which standard deviations are retrieved.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A NumPy array containing the standard deviations of unanticipated shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            std_u = obj._getv_std_u(variant)
            print(std_u)
        ```
        ................................................................................
        """
        shocks = self._invariant.dynamic_descriptor.solution_vectors.unanticipated_shocks
        return _retrieve_stds(self, variant, shocks, )

    def _getv_std_w(self, variant, /, ):
        r"""
        ................................................................................
        ==Retrieve Standard Deviations of Measurement Shocks for a Variant==

        Extracts the standard deviations of measurement shocks for a specific
        model variant. This method retrieves values from the associated solution
        vectors.

        ### Input arguments ###
        ???+ input "variant"
            The model variant for which standard deviations are retrieved.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A NumPy array containing the standard deviations of measurement shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            std_w = obj._getv_std_w(variant)
            print(std_w)
        ```
        ................................................................................
        """
        shocks = self._invariant.dynamic_descriptor.solution_vectors.measurement_shocks
        return _retrieve_stds(self, variant, shocks, )

    def _getv_cov_u(self, variant, /, ):
        r"""
        ................................................................................
        ==Compute Covariance Matrix of Unanticipated Shocks for a Variant==

        Computes the covariance matrix of unanticipated shocks for a specific
        model variant. The covariance matrix is derived from the standard 
        deviations.

        ### Input arguments ###
        ???+ input "variant"
            The model variant for which the covariance matrix is computed.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A NumPy array representing the covariance matrix of unanticipated shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            cov_u = obj._getv_cov_u(variant)
            print(cov_u)
        ```
        ................................................................................
        """
        stds = self._getv_std_u(variant, )
        return _np.diag(stds**2, )

    def _getv_cov_w(self, variant, /, ):
        r"""
        ................................................................................
        ==Compute Covariance Matrix of Measurement Shocks for a Variant==

        Computes the covariance matrix of measurement shocks for a specific
        model variant. The covariance matrix is derived from the standard 
        deviations.

        ### Input arguments ###
        ???+ input "variant"
            The model variant for which the covariance matrix is computed.

        ### Returns ###
        ???+ returns "_np.ndarray"
            A NumPy array representing the covariance matrix of measurement shocks.

        ### Example for a Method ###
        ```python
            obj = Inlay()
            cov_w = obj._getv_cov_w(variant)
            print(cov_w)
        ```
        ................................................................................
        """
        stds = self._getv_std_w(variant, )
        return _np.diag(stds**2, )


def _stack_singleton(x: tuple[_np.ndarray], /, ) -> _np.ndarray:
    return x[0]


def _stack_nonsingleton(x: tuple[_np.ndarray], /, ) -> _np.ndarray:
    return _np.dstack(x)


_STACK_FUNC_FACTORY = {
    True: _stack_singleton,
    False: _stack_nonsingleton,
}


def _get_dimension_names(
    self,
    system_vector: tuple[_incidences.Token, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    r"""
    ................................................................................
    ==Retrieve Dimension Names for System Variables==

    Generates human-readable names for rows and columns of the covariance or 
    autocovariance matrices based on the system variables.

    This utility function uses quantity names and log-linearization flags to 
    construct dimension names. These names are used for labeling the rows and 
    columns of covariance matrices.

    ### Input arguments ###
    ???+ input "self"
        The `Inlay` instance providing access to model invariants and utilities.
    ???+ input "system_vector"
        A tuple of tokens representing the system variables (transition and 
        measurement variables).

    ### Returns ###
    ???+ returns "tuple"
        A tuple containing:
        - Row names for the dimensions as a tuple of strings.
        - Column names for the dimensions as a tuple of strings.

    ### Example for a Function ###
    ```python
        obj = Inlay()
        row_names, col_names = _get_dimension_names(obj, system_vector)
        print(row_names, col_names)
    ```
    ................................................................................
    """
    qid_to_name = self.create_qid_to_name()
    qid_to_logly = self.create_qid_to_logly()
    names = tuple(
        _quantities.wrap_logly(qid_to_name[tok.qid], qid_to_logly[tok.qid], )
        for tok in system_vector
    )
    return _namings.DimensionNames(rows=names, columns=names, )


def _get_system_vector(
    self,
    /,
) -> tuple[tuple[_incidences.Token, ...], tuple[bool, ...]]:
    r"""
    ................................................................................
    ==Retrieve System Vector and Zero-Shift Boolean Mask==

    Extracts the combined vector of transition and measurement variables and 
    applies a boolean mask to filter for variables with zero shifts.

    ### Input arguments ###
    ???+ input "self"
        The `Inlay` instance.

    ### Returns ###
        ???+ returns "tuple"
        A tuple containing:
        - The filtered system vector with zero shifts.
        - A boolean mask indicating zero-shift variables.

    ### Example for a Function ###
    ```python
        obj = Inlay()
        system_vector, boolex = _get_system_vector(obj)
        print(system_vector, boolex)
    ```
    ................................................................................
    """
    system_vector = \
        self._invariant.dynamic_descriptor.solution_vectors.transition_variables \
        + self._invariant.dynamic_descriptor.solution_vectors.measurement_variables
    boolex_zero_shift = tuple(tok.shift == 0 for tok in system_vector)
    system_vector = tuple(_it.compress(system_vector, boolex_zero_shift, ))
    return system_vector, boolex_zero_shift


def _retrieve_stds(self, variant, shocks, ) -> _np.ndarray:
    r"""
    ................................................................................
    ==Retrieve Standard Deviations from a Model Variant==

    A utility function that retrieves the standard deviations associated with 
    specific shocks in a model variant.

    ### Input arguments ###
    ???+ input "self"
        The `Inlay` instance.
    ???+ input "variant"
        The model variant from which standard deviations are retrieved.
    ???+ input "shocks"
        The shocks for which standard deviations are retrieved.

    ### Returns ###
    ???+ returns "_np.ndarray"
        A NumPy array containing the standard deviations for the specified shocks.

    ### Example for a Function ###
    ```python
        obj = Inlay()
        stds = _retrieve_stds(obj, variant, shocks)
        print(stds)
    ```
    ................................................................................
    """
    #[
    std_qids = tuple(
        self._invariant.shock_qid_to_std_qid[t.qid]
        for t in shocks
    )
    return variant.retrieve_values_as_array("levels", std_qids, )
    #]

