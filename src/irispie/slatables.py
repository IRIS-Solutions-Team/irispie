"""
Slatable protocol
"""


#[
from __future__ import annotations
#]


class Slatable:
    """
    """
    #[

    __slots__ = (
        #
        "shocks_from_data",
        "stds_from_data",
        #
        "max_lag",
        "max_lead",
        #
        "databox_names",
        "databox_validators",
        "output_names",
        #
        "fallbacks",
        "overwrites",
        "qid_to_logly",
    )

    def __init__(
        self,
        *,
        shocks_from_data: bool = False,
        stds_from_data: bool = False,
        parameters_from_data: bool = False,
        **kwargs,
    ) -> None:
        """
        """
        self.shocks_from_data = shocks_from_data
        self.stds_from_data = stds_from_data
        self.parameters_from_data = parameters_from_data
        #
        self.max_lag = None
        self.max_lead = None
        self.databox_names = None
        self.databox_validators = None
        self.fallbacks = None
        self.overwrites = None
        self.qid_to_logly = None
        self.output_names = None

    #]

