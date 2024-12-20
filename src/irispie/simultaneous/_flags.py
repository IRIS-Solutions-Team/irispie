"""
"""
#[
from __future__ import annotations

import enum as _en
#]


class Flags(_en.IntFlag, ):
    r"""
    ................................................................................
    ==Flags: A Set of IntFlags for Model Characteristics==

    The `Flags` class inherits from Python's `enum.IntFlag` and provides 
    a flexible way to represent and manipulate model characteristics, such 
    as linearity, flatness, and determinism. These flags allow for bitwise 
    operations and are used to encode various state configurations.

    Flags available:
    - `DEFAULT`: Represents no flags set.
    - `LINEAR`: Indicates that the model is linear.
    - `FLAT`: Indicates that the model operates in a flat structure.
    - `DETERMINISTIC`: Indicates that the model is deterministic.

    Methods and properties in this class allow querying and updating flags 
    efficiently, including utilities to initialize flags from keyword arguments.

    ................................................................................
    """

    #[

    DEFAULT = 0
    LINEAR = _en.auto()
    FLAT = _en.auto()
    DETERMINISTIC = _en.auto()

    @property
    def is_linear(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Linear Flag is Set==

        Returns whether the `LINEAR` flag is set for this instance.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `LINEAR` flag is set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.LINEAR
            print(flags.is_linear)  # Output: True
        ```
        ................................................................................
        """
        return Flags.LINEAR in self

    @property
    def is_nonlinear(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Model is Nonlinear==

        Returns whether the `LINEAR` flag is not set, implying the model is nonlinear.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `LINEAR` flag is not set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.DEFAULT
            print(flags.is_nonlinear)  # Output: True
        ```
        ................................................................................
        """
        return not self.is_linear

    @property
    def is_flat(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Flat Flag is Set==

        Returns whether the `FLAT` flag is set for this instance.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `FLAT` flag is set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.FLAT
            print(flags.is_flat)  # Output: True
        ```
        ................................................................................
        """
        return Flags.FLAT in self

    @property
    def is_nonflat(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Model is Non-Flat==

        Returns whether the `FLAT` flag is not set, implying the model is non-flat.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `FLAT` flag is not set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.DEFAULT
            print(flags.is_nonflat)  # Output: True
        ```
        ................................................................................
        """
        return not self.is_flat

    @property
    def is_deterministic(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Deterministic Flag is Set==

        Returns whether the `DETERMINISTIC` flag is set for this instance.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `DETERMINISTIC` flag is set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.DETERMINISTIC
            print(flags.is_deterministic)  # Output: True
        ```
        ................................................................................
        """
        return Flags.DETERMINISTIC in self

    @property
    def is_stochastic(self, /, ) -> bool:
        r"""
        ................................................................................
        ==Check if the Model is Stochastic==

        Returns whether the `DETERMINISTIC` flag is not set, implying the model 
        is stochastic.

        ### Input arguments ###
        ???+ input "None"
            This property does not take any input arguments.

        ### Returns ###
        ???+ returns "bool"
            `True` if the `DETERMINISTIC` flag is not set; otherwise, `False`.

        ### Example for a Property ###
        ```python
            flags = Flags.DEFAULT
            print(flags.is_stochastic)  # Output: True
        ```
        ................................................................................
        """
        return not self.is_deterministic

    def update_from_kwargs(self, /, **kwargs) -> Self:
        r"""
        ................................................................................
        ==Update Flags Using Keyword Arguments==

        Updates the current flags based on the provided keyword arguments. If a 
        flag is not specified in the arguments, its existing state is retained.

        ### Input arguments ###
        ???+ input "**kwargs"
            Keyword arguments specifying flag states. Possible keys are:
            - `linear`: Updates the `LINEAR` flag.
            - `flat`: Updates the `FLAT` flag.
            - `deterministic`: Updates the `DETERMINISTIC` flag.

        ### Returns ###
        ???+ returns "Self"
            A new `Flags` instance with updated flags.

        ### Example for a Method ###
        ```python
            flags = Flags.DEFAULT
            updated_flags = flags.update_from_kwargs(linear=True, flat=False)
            print(updated_flags)
        ```
        ................................................................................
        """
        linear = kwargs.get("linear") if kwargs.get("linear") is not None else self.is_linear
        flat = kwargs.get("flat") if kwargs.get("flat") is not None else self.is_flat
        deterministic = kwargs.get("deterministic") if kwargs.get("deterministic") is not None else self.is_deterministic
        return type(self).from_kwargs(linear=linear, flat=flat, deterministic=deterministic, )

    @classmethod
    def from_kwargs(cls: type, **kwargs, ) -> Self:
        r"""
        ................................................................................
        ==Create a Flags Instance from Keyword Arguments==

        Initializes a `Flags` instance by setting the flags specified in the 
        provided keyword arguments.

        ### Input arguments ###
        ???+ input "**kwargs"
            Keyword arguments specifying flag states. Possible keys are:
            - `linear`: Sets the `LINEAR` flag.
            - `flat`: Sets the `FLAT` flag.
            - `deterministic`: Sets the `DETERMINISTIC` flag.

        ### Returns ###
        ???+ returns "Self"
            A new `Flags` instance with the specified flags set.

        ### Example for a Method ###
        ```python
            flags = Flags.from_kwargs(linear=True, deterministic=False)
            print(flags)
        ```
        ................................................................................
        """
        self = cls.DEFAULT
        if kwargs.get("linear"):
            self |= cls.LINEAR
        if kwargs.get("flat"):
            self |= cls.FLAT
        if kwargs.get("deterministic"):
            self |= cls.DETERMINISTIC
        return self

    #]
