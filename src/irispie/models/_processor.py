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
    def set_preprocessor(
        self,
        preprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        self._invariant.preprocessor = preprocessor

    def set_postprocessor(
        self,
        postprocessor: _sequentials.Sequential,
        /,
    ) -> None:
        self._invariant.postprocessor = postprocessor

    def preprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        return self._invariant.preprocessor.simulate(*args, **kwargs, )

    def postprocess(
        self,
        *args,
        **kwargs,
    ) -> tuple:
        return self._invariant.postprocessor.simulate(*args, **kwargs, )
    #]

