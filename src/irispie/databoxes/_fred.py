"""
"""


#[
from __future__ import annotations

from typing import (Iterable, )
import requests as _rq

from . import main as _databoxes
from .. import dates as _dates
from ..series import main as _series
#]


_FRED_FREQ_MAP = {
    "A".casefold(): _dates.Freq.YEARLY,
    "Q".casefold(): _dates.Freq.QUARTERLY,
    "M".casefold(): _dates.Freq.MONTHLY,
    "D".casefold(): _dates.Freq.DAILY,
}


_BASE_URL = r"https://api.stlouisfed.org/fred/series"
_OBSERVATIONS_URL = _BASE_URL + r"/observations"
_API_KEY = r"951f01181da86ccb9045ce8716f82f43"
_PARAMETERS = r"?series_id={series_id}&api_key={api_key}&file_type=json"
_MISSING_VALUE = r"."


class FredMixin:
    """
    """
    #[

    @classmethod
    def from_fred(
        klass,
        mapper: Iterable[str] | dict[str, str],
        /,
    ) -> _databoxes.Databox:
        """
        """
        self = klass()
        if not isinstance(mapper, dict):
            mapper = _mapper_from_series_ids(mapper, )
        for name, series_id in mapper.items():
            self[name] = _get_series(series_id, )
        return self

    #]


def _mapper_from_series_ids(series_ids: Iterable[str], /, ):
    """
    """
    #[
    return {
        series_id.strip(): series_id.strip()
        for series_id in series_ids
    }
    #]

def _get_series(series_id: str, /, ):
    """
    """
    #[
    urls = _get_series_urls(series_id, )
    meta_response = _rq.get(urls["meta_url"], ).json()
    data_response = _rq.get(urls["data_url"], ).json()
    #
    if "seriess" not in meta_response or "observations" not in data_response:
        raise IriePieCritical(f"Invalid response from FRED API for {series_id}", )
    #
    freq = _get_freq_from_meta_response(meta_response, )
    iso_dates, str_values = _get_dates_and_values_from_data_response(data_response, )
    values = ( (float(x) if x != _MISSING_VALUE else None) for x in str_values )
    dates = _dates.daters_from_iso_strings(freq, iso_dates, )
    return _series.Series(dates=dates, values=list(values), )
    #]


def _get_freq_from_meta_response(meta_response: dict, /, ):
    """
    """
    #[
    freq_letter = meta_response["seriess"][0]["frequency_short"].casefold()
    return _FRED_FREQ_MAP[freq_letter]
    #]


def _get_dates_and_values_from_data_response(data_response: dict, /, ):
    """
    """
    #[
    date_value_pairs = (
        (obs["date"], obs["value"], )
        for obs in data_response["observations"]
    )
    return zip(*date_value_pairs, )
    #]


def _get_series_urls(series_id: str, /, ):
    """
    """
    #[
    parameters = _PARAMETERS.format(series_id=series_id, api_key=_API_KEY, )
    meta_url = _BASE_URL + parameters
    data_url = _OBSERVATIONS_URL + parameters
    return {"meta_url": meta_url, "data_url": data_url, }
    #]

