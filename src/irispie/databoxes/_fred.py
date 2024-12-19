"""
Fred API Integration Module
"""


#[
from __future__ import annotations

from typing import (Iterable, )
import requests as _rq

from .. import dates as _dates
from ..dates import (Frequency, )
from ..series.main import (Series, )

from . import main as _databoxes
#]


_FRED_FREQ_MAP = {
    "A".casefold(): Frequency.YEARLY,
    "Q".casefold(): Frequency.QUARTERLY,
    "M".casefold(): Frequency.MONTHLY,
    "D".casefold(): Frequency.DAILY,
}


_BASE_URL = r"https://api.stlouisfed.org/fred/series"
_OBSERVATIONS_URL = _BASE_URL + r"/observations"
_API_KEY = r"951f01181da86ccb9045ce8716f82f43"
_PARAMETERS = r"?series_id={series_id}&api_key={api_key}&file_type=json"
_MISSING_VALUE = r"."


class FredMixin:
    r"""
    ................................................................................
    ==FRED Mixin for Data Import==

    Provides methods to load and manage data from the Federal Reserve Economic 
    Data (FRED) API. Supports automatic mapping of series names to FRED series IDs.

    This mixin facilitates the creation of `Databox` objects containing time series 
    data retrieved from the FRED API.

    ### Example ###
    ```python
        class MyDataBox(FredMixin, _databoxes.Databox):
            pass

        databox = MyDataBox.from_fred(["UNRATE", "GDP"])
    ```
    ................................................................................
    """
    #[

    @classmethod
    def from_fred(
        klass,
        mapper: Iterable[str] | dict[str, str],
        /,
    ) -> _databoxes.Databox:
        r"""
        ................................................................................
        ==Create a Databox from FRED Data==

        Load time series data from the FRED API into a `Databox`. Accepts a list of 
        series IDs or a mapping of custom names to series IDs.

        ### Input arguments ###
        ???+ input "klass"
            The class invoking this method (typically a `Databox` subclass).

        ???+ input "mapper"
            An iterable of series IDs or a dictionary mapping custom names to series IDs.

        ### Returns ###
        ???+ returns
            `_databoxes.Databox`: A `Databox` populated with data retrieved from FRED.

        ### Example ###
        ```python
            databox = MyDataBox.from_fred(["UNRATE", "CPIAUCSL"])
            print(databox)
        ```
        ................................................................................
        """
        self = klass()
        if not isinstance(mapper, dict):
            mapper = _mapper_from_series_ids(mapper, )
        for name, series_id in mapper.items():
            self[name] = _get_series(series_id, )
        return self

    #]


def _mapper_from_series_ids(series_ids: Iterable[str], /, ):
    r"""
    ................................................................................
    ==Generate Mapper from Series IDs==

    Create a mapping of series IDs to themselves for use with `from_fred`.

    ### Input arguments ###
    ???+ input "series_ids"
        An iterable of series IDs.

    ### Returns ###
    ???+ returns
        `dict[str, str]`: A dictionary mapping each series ID to itself.

    ### Example ###
    ```python
        mapper = _mapper_from_series_ids(["UNRATE", "GDP"])
        print(mapper)  # Output: {"UNRATE": "UNRATE", "GDP": "GDP"}
    ```
    ................................................................................
    """
    #[
    return {
        series_id.strip(): series_id.strip()
        for series_id in series_ids
    }
    #]

def _get_series(series_id: str, /, ):
    r"""
    ................................................................................
    ==Retrieve Series Data from FRED==

    Fetch metadata and observations for a given series ID from the FRED API.

    ### Input arguments ###
    ???+ input "series_id"
        The FRED series ID to retrieve data for.

    ### Returns ###
    ???+ returns
        `Series`: A `Series` object containing the data for the given ID.

    ### Example ###
    ```python
        series = _get_series("UNRATE")
        print(series)
    ```
    ................................................................................
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
    dates = _dates.periods_from_iso_strings(iso_dates, frequency=freq, )
    return Series(dates=dates, values=list(values), )
    #]


def _get_freq_from_meta_response(meta_response: dict, /, ):
    r"""
    ................................................................................
    ==Extract Frequency from Metadata==

    Retrieve the frequency of the series from the FRED API metadata response.

    ### Input arguments ###
    ???+ input "meta_response"
        The JSON metadata response from the FRED API.

    ### Returns ###
    ???+ returns
        `Frequency`: The frequency of the series.

    ### Example ###
    ```python
        freq = _get_freq_from_meta_response(meta_response)
        print(freq)  # Output: Frequency.MONTHLY
    ```
    ................................................................................
    """
    #[
    freq_letter = meta_response["seriess"][0]["frequency_short"].casefold()
    return _FRED_FREQ_MAP[freq_letter]
    #]


def _get_dates_and_values_from_data_response(data_response: dict, /, ):
    r"""
    ................................................................................
    ==Extract Dates and Values from Data Response==

    Parse the observations section of the FRED API data response to extract 
    dates and corresponding values.

    ### Input arguments ###
    ???+ input "data_response"
        The JSON data response from the FRED API.

    ### Returns ###
    ???+ returns
        `tuple[Iterable[str], Iterable[str]]`: A tuple of two iterables - 
        one for the dates and another for the values.

    ### Example ###
    ```python
        dates, values = _get_dates_and_values_from_data_response(data_response)
        print(list(dates), list(values))
    ```
    ................................................................................
    """
    #[
    date_value_pairs = (
        (obs["date"], obs["value"], )
        for obs in data_response["observations"]
    )
    return zip(*date_value_pairs, )
    #]


def _get_series_urls(series_id: str, /, ):
    r"""
    ................................................................................
    ==Generate URLs for FRED Series==

    Construct metadata and data retrieval URLs for a given series ID using 
    the FRED API base URLs and parameters.

    ### Input arguments ###
    ???+ input "series_id"
        The FRED series ID for which to construct URLs.

    ### Returns ###
    ???+ returns
        `dict[str, str]`: A dictionary containing:
        - `meta_url`: URL to fetch metadata for the series.
        - `data_url`: URL to fetch data observations for the series.

    ### Example ###
    ```python
        urls = _get_series_urls("UNRATE")
        print(urls["meta_url"])  # Metadata URL
        print(urls["data_url"])  # Data URL
    ```
    ................................................................................
    """
    #[
    parameters = _PARAMETERS.format(series_id=series_id, api_key=_API_KEY, )
    meta_url = _BASE_URL + parameters
    data_url = _OBSERVATIONS_URL + parameters
    return {"meta_url": meta_url, "data_url": data_url, }
    #]

