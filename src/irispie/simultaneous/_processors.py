"""
Provides a mixin for managing preprocessor and postprocessor sequential operations.
"""


#[
from __future__ import annotations

from ..sequentials import main as _sequentials
#]


class ProcessorMixin:
    r"""
    ................................................................................
    ==Class: ProcessorMixin==

    Adds preprocessor and postprocessor functionality to a class. This mixin manages 
    sequential operations for pre- and post-processing tasks, allowing simulation 
    workflows to be easily integrated.

    Attributes:
        - `preprocessor`: A sequential object handling preprocessing tasks.
        - `postprocessor`: A sequential object handling postprocessing tasks.
    ................................................................................
    """
    #[

    @property
    def preprocessor(self, /, ) -> _sequentials.Sequential:
        r"""
        ................................................................................
        ==Property: preprocessor==

        Gets the preprocessor associated with this instance. The preprocessor is a 
        sequential object responsible for handling preprocessing tasks in a simulation 
        workflow.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "_sequentials.Sequential"
            The preprocessor object.

        ### Example ###
        ```python
            preproc = obj.preprocessor
        ```
        ................................................................................
        """
        return self._invariant.preprocessor

    @property
    def postprocessor(self, /, ) -> _sequentials.Sequential:
        r"""
        ................................................................................
        ==Property: postprocessor==

        Gets the postprocessor associated with this instance. The postprocessor is a 
        sequential object responsible for handling postprocessing tasks in a simulation 
        workflow.

        ### Input arguments ###
        (No arguments)

        ### Returns ###
        ???+ returns "_sequentials.Sequential"
            The postprocessor object.

        ### Example ###
        ```python
            postproc = obj.postprocessor
        ```
        ................................................................................
        """
        return self._invariant.postprocessor

    @preprocessor.setter
    def preprocessor(
        self,
        preprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        r"""
        ................................................................................
        ==Property Setter: preprocessor==

        Sets the preprocessor for this instance. The preprocessor is used to handle 
        preprocessing tasks in simulation workflows.

        ### Input arguments ###
        ???+ input "preprocessor: _sequentials.Sequential"
            The preprocessor object to associate with this instance.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.preprocessor = new_preprocessor
        ```
        ................................................................................
        """
        self._invariant.preprocessor = preprocessor

    @postprocessor.setter
    def postprocessor(
        self,
        postprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        r"""
        ................................................................................
        ==Property Setter: postprocessor==

        Sets the postprocessor for this instance. The postprocessor is used to handle 
        postprocessing tasks in simulation workflows.

        ### Input arguments ###
        ???+ input "postprocessor: _sequentials.Sequential"
            The postprocessor object to associate with this instance.

        ### Returns ###
        (No return value)

        ### Example ###
        ```python
            obj.postprocessor = new_postprocessor
        ```
        ................................................................................
        """
        self._invariant.postprocessor = postprocessor

    def preprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        r"""
        ................................................................................
        ==Method: preprocess==

        Executes the preprocessor's simulation tasks using the provided arguments. This 
        method processes input data before the main simulation.

        ### Input arguments ###
        ???+ input "*args"
            Positional arguments to pass to the preprocessor.
        ???+ input "**kwargs"
            Keyword arguments to pass to the preprocessor.

        ### Returns ###
        ???+ returns "tuple"
            The result of the preprocessor's simulation.

        ### Example ###
        ```python
            result = obj.preprocess(arg1, arg2, key=value)
        ```
        ................................................................................
        """
        return self.preprocessor.simulate(*args, **kwargs, )

    def postprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        r"""
        ................................................................................
        ==Method: postprocess==

        Executes the postprocessor's simulation tasks using the provided arguments. This 
        method processes output data after the main simulation.

        ### Input arguments ###
        ???+ input "*args"
            Positional arguments to pass to the postprocessor.
        ???+ input "**kwargs"
            Keyword arguments to pass to the postprocessor.

        ### Returns ###
        ???+ returns "tuple"
            The result of the postprocessor's simulation.

        ### Example ###
        ```python
            result = obj.postprocess(arg1, arg2, key=value)
        ```
        ................................................................................
        """
        return self.postprocessor.simulate(*args, **kwargs, )

    #]

