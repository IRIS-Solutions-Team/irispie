r"""
Reduced-form vector autoregression models
"""


#[

from __future__ import annotations

from typing import NoReturn

from .. import has_invariant as _has_invariant
from .. import has_variants as _has_variants
from .. import quantities as _quantities
from ..quantities import Quantity, QuantityKind
from ..fords import covariances as _covariances
from ..namings import DimensionNames

from . import _estimators as _estimators
from . import _simulators as _simulators
from . import _slatable_protocols as _slatable_protocols
from ._invariants import Invariant
from ._variants import Variant, System
from ._dimensions import Dimensions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
    from collections.abc import Iterable
    from ..fords.descriptors import Solution, SolutionVectors

#]


__all__ = (
    "RedVAR",
)


class RedVAR(
    _estimators.Inlay,
    _simulators.Inlay,
    _slatable_protocols.Inlay,
    _has_invariant.Mixin,
    _has_variants.Mixin,
    _quantities.Mixin,
):
    r"""
    """
    #[

    __slots__ = (
        "_invariant",
        "_variants",
    )

    max_lead = 0

    def __init__(
        self,
        *args,
        num_variants: int = 1,
        **kwargs,
    ) -> None | NoReturn:
        """
        """
        if not args and not kwargs:
            self._invariant = None
            self._variants = []
            return
        self._invariant = Invariant(*args, **kwargs, )
        self._variants = [ Variant() for _ in range(num_variants, ) ]

    @classmethod
    def skeleton(
        klass,
        other: Self,
    ) -> Self:
        r"""
        """
        self = klass()
        self._invariant = other._invariant
        self._variants = []
        return self

    def copy(self, ) -> Self:
        r"""
        """
        new = type(self)()
        new._invariant = self._invariant
        new._variants = [ v.copy() for v in self._variants ]
        return new

    def get_endogenous_names(self, ) -> tuple[str, ...]:
        r"""
        """
        quantities = self._access_quantities()
        return tuple(_quantities.generate_names_of_kind(quantities, QuantityKind.TRANSITION_VARIABLE, ))

    def get_exogenous_names(self, ) -> tuple[str, ...]:
        r"""
        """
        quantities = self._access_quantities()
        return tuple(_quantities.generate_names_of_kind(quantities, QuantityKind.EXOGENOUS_VARIABLE, ))

    def get_residual_names(self, ) -> tuple[str, ...]:
        r"""
        """
        quantities = self._access_quantities()
        return tuple(_quantities.generate_names_of_kind(quantities, QuantityKind.UNANTICIPATED_SHOCK, ))

    def get_endogenous_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._invariant.get_endogenous_qids()

    def get_residual_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._invariant.get_residual_qids()

    def get_exogenous_qids(self, ) -> tuple[int, ...]:
        r"""
        """
        return self._invariant.get_exogenous_qids()

    def get_system_matrices(
        self,
        unpack_singleton: bool = True,
    ) -> System | tuple[System, ...]:
        r"""
        """
        system_matrices = [ v.system for v in self._variants ]
        return self.unpack_singleton(system_matrices, unpack_singleton=unpack_singleton, )

    def get_acov_dimension_names(self, ) -> DimensionNames:
        r"""
        """
        endogenous_names = self.get_endogenous_names()
        dimension_names = DimensionNames(rows=endogenous_names, columns=endogenous_names, )
        return dimension_names

    def get_acov(
        self,
        up_to_order: int = 0,
        unpack_singleton: bool = True,
    ) -> tuple[_np.ndarray, ...] | list[tuple[_np.ndarray, ...]]:
        r"""
        """
        acov_by_variant = [
            v.get_acov(up_to_order=up_to_order, )
            for v in self._variants
        ]
        return self.unpack_singleton(
            acov_by_variant,
            unpack_singleton=unpack_singleton,
        )

    def get_acorr(self, *args, **kwargs, ):
        r"""
        """
        return _covariances.get_acorr_by_variant(self, *args, **kwargs, )

    def get_mean(
        self,
        unpack_singleton: bool = True,
    ) -> _np.ndarray | list[_np.ndarray]:
        r"""
        """
        amean_by_variant = [ v.get_mean() for v in self._variants ]
        return self.unpack_singleton(
            amean_by_variant,
            unpack_singleton=unpack_singleton,
        )

    @property
    def dimensions(self, ) -> Dimensions:
        r"""
        """
        return self._invariant.dimensions

    @property
    def order(self, ) -> int:
        r"""
        """
        return self._invariant.dimensions.order

    @property
    def max_lag(self, ) -> int:
        r"""
        """
        return self._invariant.dimensions.order

    @property
    def has_intercept(self, ) -> bool:
        r"""
        """
        return self._invariant.dimensions.has_intercept

    @property
    def has_exogenous(self, ) -> int:
        r"""
        """
        return self._invariant.dimensions.has_exogenous

    def _get_dynamic_solution_vectors(self, ) -> SolutionVectors:
        r"""
        """
        return self._invariant.solution_vectors

    def _gets_solution(
        self,
        vid: int = 0,
        **kwargs,
    ) -> Solution:
        r"""
        """
        return self._variants[0]._get_companion_solution(**kwargs, )

    def get_companion_matrices(
        self,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> tuple[System, ...] | System:
        r"""
        """
        companion_matrices = [
            v._get_companion_solution(**kwargs, )
            for v in self._variants
        ]
        return self.unpack_singleton(
            companion_matrices,
            unpack_singleton=unpack_singleton,
        )

    def get_stability(
        self,
        unpack_singleton: bool = True,
    ) -> list[bool, ...] | bool:
        r"""
        """
        stability = [ v.is_stable for v in self._variants ]
        return self.unpack_singleton(
            stability,
            unpack_singleton=unpack_singleton,
        )

    def get_max_abs_eigenvalue(
        self,
        unpack_singleton: bool = True,
    ) -> list[bool, ...] | bool:
        r"""
        """
        max_abs_eigenvalue = [ v.max_abs_eigenvalue for v in self._variants ]
        return self.unpack_singleton(
            max_abs_eigenvalue,
            unpack_singleton=unpack_singleton,
        )

    def get_eigenvalues(
        self,
        unpack_singleton: bool = True,
    ) -> list[bool, ...] | bool:
        r"""
        """
        eigenvalues = [ v.eigenvalues for v in self._variants ]
        return self.unpack_singleton(
            eigenvalues,
            unpack_singleton=unpack_singleton,
        )

#]

