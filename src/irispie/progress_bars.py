r"""
"""


#[

from __future__ import annotations

import time as _tm
import sys
from typing import Any

#]


__all__ = (
    "ProgressBar",
)


_DEFAULT_EMPTY_CHAR = "◻︎"
_DEFAULT_FILL_CHAR = "◼︎"
_DEFAULT_BAR_LENGTH = 50


class ProgressBar:
    r"""
    """

    def __init__(
        self,
        num_steps: int,
        #
        title: str = "",
        bar_length: int = _DEFAULT_BAR_LENGTH,
        fill_char: str = _DEFAULT_FILL_CHAR,
        empty_char: str = _DEFAULT_EMPTY_CHAR,
        show_progress: bool = True,
        progress_format: str = "{title} {bar} {percent:.2f}%",
        final_format: str = "{title} {bar} {time_elapsed:.4f}s",
        output_stream: Any = sys.stdout,
        start: bool = True,
        finish_when_complete: bool = True,
    ) -> None:
        r"""
        """
        self.num_steps = num_steps
        self.title = title
        self.bar_length = bar_length
        self.current_step = 0 
        self.fill_char = fill_char[0]
        self.empty_char = empty_char[0]
        self.show_progress = show_progress
        self.progress_format = progress_format
        self.final_format = final_format
        self.output_stream = output_stream
        self.time_started = None
        self.time_finished = None
        self.finish_when_complete = finish_when_complete
        if start:
            self.start()

    @property
    def is_complete(self) -> bool:
        r"""
        """
        return self.current_step >= self.num_steps

    def start(self, ) -> None:
        r"""
        """
        self.time_started = _tm.time()
        self.update(0, )

    def update(self, step: int, ) -> None:
        r"""
        """
        self.current_step = step
        if self.show_progress:
            self._print_bar()
        if self.finish_when_complete and self.is_complete:
            self.finish()

    def increment(self, step: int = 1, ) -> None:
        r"""
        """
        self.update(self.current_step + step, )

    def finish(self, ) -> None:
        r"""
        """
        self.time_finished = _tm.time()
        if self.show_progress:
            self._print_final()

    def _print_bar(self, ) -> None:
        r"""
        """
        context = self._eval_context(_tm.time(), )
        progress_line = self.progress_format.format(**context, )
        print(progress_line, end="\r", file=self.output_stream, flush=True, )

    def _print_final(self, ) -> None:
        r"""
        """
        context = self._eval_context(self.time_finished, )
        final_line = self.final_format.format(**context, )
        print(final_line, file=self.output_stream, flush=True, )

    def _eval_context(self, time, ) -> dict[str, Any]:
        r"""
        """
        filled_length = int(self.bar_length * self.current_step // self.num_steps)
        return dict(
            title=self.title,
            percent=(self.current_step / self.num_steps) * 100,
            bar=self.fill_char * filled_length + self.empty_char * (self.bar_length - filled_length),
            time_elapsed=time - self.time_started,
        )

