"""
"""


#[
from __future__ import annotations

from ..sequentials import main as _sequentials
#]


class ProcessorMixin:
    """
    """
    #[

    @property
    def preprocessor(self, /, ) -> _sequentials.Sequential:
        """
        """
        return self._invariant.preprocessor

    @property
    def postprocessor(self, /, ) -> _sequentials.Sequential:
        """
        """
        return self._invariant.postprocessor

    @preprocessor.setter
    def preprocessor(
        self,
        preprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        """
        """
        self._invariant.preprocessor = preprocessor

    @postprocessor.setter
    def postprocessor(
        self,
        postprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        """
        """
        self._invariant.postprocessor = postprocessor

    def preprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        return self.preprocessor.simulate(*args, **kwargs, )

    def postprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        return self.postprocessor.simulate(*args, **kwargs, )

    #]

