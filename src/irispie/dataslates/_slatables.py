r"""
Slatable protocol
"""


#[

from __future__ import annotations

#]


class Slatable:
    r"""
    """
    #[

    max_lag: int = 0
    max_lead: int = 0
    databox_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    fallbacks: dict[str, Real] | None = None
    overwrites: dict[str, Real] | None = None
    qid_to_logly: dict[str, bool] | None = None
    databox_validators: dict[str, Callable] | None = None
    descriptions: tuple[str, ...] = ()

    #]

