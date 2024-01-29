"""
Interface to X13-ARIMA-TRAMO-SEATS
"""


#[
from __future__ import annotations
from typing import (Any, )
from types import (EllipsisType, )
import os as _os
import numpy as _np
import subprocess as _sp
import platform as _pf
import tempfile as _tf
import glob as _gl

from .. import executables as _executables
from .. import wrongdoings as _wrongdoings
from .. import dates as _dates
from . import main as _series
from . import _functionalize
#]


__all__ = ()


class Inlay:
    """
    """
    #[
    def x13(
        self,
        *,
        range: _dates.Ranger | EllipsisType = ...,
        when_error: _wrongdoings.HOW = "warning",
        clean_up: bool = True,
        output: str = "sa",
        mode: Literal["mult", "add", "pseudoadd", "logadd", ] = "mult",
        **kwargs,
    ) -> dict[str, Any]:
        """
        """
        range = tuple(self._resolve_dates(range))
        base_start_date = range[0]
        settings = _create_settings(base_start_date, output, mode, **kwargs, )
        spc = _create_spc_file(_TEMPLATE_SPC, settings, )
        new_data = []
        info = []
        for variant_data in self.data.T:
            ith_new_data, ith_info = _x13_data(spc, base_start_date, variant_data, settings["x11_save"], clean_up=clean_up, )
            new_data.append(ith_new_data.reshape(-1, 1, ))
            info.append(ith_info)
        if not all(i["success"] for i in info):
            _wrongdoings.raise_as(
                when_error,
                "X13 failed to produce a result for at least one variant.",
            )
        new_data = _np.hstack(new_data, )
        self._replace_start_date_and_values(base_start_date, new_data, )
        return info
    #]


for n in ("x13", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _x13_data(
    spc: str,
    base_start_date: _dates.Date,
    base_variant_data: _np.ndarray,
    x11_save: str,
    /,
    clean_up: bool = True,
) -> tuple[_np.ndarray, dict[str, Any]]:
    """
    """
    #[
    new_data = _np.full(base_variant_data.shape, _np.nan, dtype=float, )
    start_date, variant_data, output_slice = _remove_leading_trailing_nans(base_start_date, base_variant_data, )
    year, period = start_date.to_year_period()
    spc = spc.replace("$(series_start)", _X13_DATE_FORMAT_RESOLUTION[start_date.frequency].format(year=year, period=period, ), )
    spc = spc.replace("$(series_data)", _print_series_data(variant_data, ), )
    spc_file_name_without_ext = _write_spc_to_file(spc, )
    system_output = _execute(spc_file_name_without_ext, )
    raw_output_data, info = _collect_outputs(spc_file_name_without_ext, system_output, x11_save, )
    if raw_output_data is not None:
        new_data[output_slice] = raw_output_data
    if clean_up:
        _clean_up(spc_file_name_without_ext, )
    return new_data, info
    #]


def _print_series_data(
    variant_data: _np.ndarray,
    /,
) -> str:
    """
    """
    return "\n".join(f"        {v:g}" for v in variant_data)


def _execute(
    spc_file_name_without_ext: str,
    /,
) -> _sp.CompletedProcess:
    """
    """
    #[
    return _sp.run(
        [_X13_EXECUTABLE_PATH, spc_file_name_without_ext, ],
        stdout=_sp.PIPE,
        check=True,
    )
    #]


def _collect_outputs(
    spc_file_name_without_ext: str,
    output: _sp.CompletedProcess,
    x11_save: str,
    /,
) -> tuple[_np.ndarray | None, dict[str, str]]:
    """
    """
    #[
    info = _read_output_info(spc_file_name_without_ext, )
    info["success"] = output.returncode == 0 and "ERROR" not in str(output.stdout)
    if info["success"]:
        output_data, info[x11_save] = _read_output_data(spc_file_name_without_ext, x11_save, info, )
    else:
        output_data = None
        info[x11_save] = None
    return output_data, info
    #]


def _read_output_data(
    spc_file_name_without_ext: str,
    x11_save: str,
    info: dict[str, str],
    /,
) -> tuple[tuple[float, ...], str]:
    """
    """
    #[
    with open(spc_file_name_without_ext + "." + x11_save, "rt", ) as fid:
        table = str(fid.read())
    output_data = tuple(float(i) for i in table.split()[5::2])
    return output_data, table
    #]


def _read_output_info(
    spc_file_name_without_ext: str,
    /,
) ->  dict[str, str]:
    """
    """
    info = dict()
    for ext in ["log", "out", "err", ]:
        with open(spc_file_name_without_ext + "." + ext, "rt", ) as fid:
            info[ext] = fid.read()
    return info


def _write_spc_to_file(
    spc: str,
    /,
) -> str:
    """
    """
    with _tf.NamedTemporaryFile(dir=".", mode="wt", suffix=".spc", delete=False, ) as fid:
        spc_file_name = fid.name
        fid.write(spc)
    return spc_file_name.removesuffix(".spc")


def _remove_leading_trailing_nans(
    base_start_date: _dates.Date,
    variant_data: _np.ndarray,
    /,
) -> tuple[_dates.Date, _np.ndarray, slice]:
    """
    """
    #[
    start_date = base_start_date.copy()
    num_rows = variant_data.shape[0]
    num_leading = 0
    num_trailing = 0
    while _np.isnan(variant_data[0]):
        num_leading += 1
        variant_data = variant_data[1:]
    while _np.isnan(variant_data[-1]):
        num_trailling += 1
        variant_data = variant_data[:-1]
    return start_date + num_leading, variant_data, slice(num_leading, num_rows-num_trailing, )
    #]


def _create_settings(
    base_start_date: _dates.Date,
    output: str,
    mode: Literal["mult", "add", "pseudoadd", "logadd", ],
    /,
    transform_function: str = "none",
) ->  dict[str, str]:
    """
    """
    #[
    if str(mode) == "mult" and str(transform_function) == "none":
        transform_function = "log"
    settings = {
        "series_period": str(base_start_date.frequency.value),
        "x11_mode": str(mode),
        "x11_save": _X11_OUTPUT_RESOLUTION.get(output, output),
        "transform_function": str(transform_function),
    }
    return settings
    #]


def _create_spc_file(
    spc: str,
    settings: dict[str, str],
    /,
) -> str:
    """
    Replace placeholders with custom setting in the spc file
    """
    #[
    for k, v in settings.items():
        spc = spc.replace(f"$({k})", v)
    return spc
    #]


def _clean_up(
    spc_file_name_without_ext: str,
    /,
) -> None:
    """
    """
    #[
    for f in _gl.glob(spc_file_name_without_ext + ".*"):
        _os.remove(f)
    #]


_EXECUTABLES_PATH = _os.path.dirname(_executables.__file__)
_TEMPLATE_SPC_PATH = _os.path.join(_EXECUTABLES_PATH, "template.spc", )


with open(_TEMPLATE_SPC_PATH, "rt", ) as fid:
    _TEMPLATE_SPC = fid.read()


_X11_OUTPUT_RESOLUTION = {
    "sf": "d10",
    "sa": "d11",
    "tc": "d12",
    "irr": "d13",
}


_X13_DATE_FORMAT_RESOLUTION = {
    _dates.Frequency.MONTHLY: "{year:04d}.{period:02d}",
    _dates.Frequency.QUARTERLY: "{year:04d}.{period:1d}",
}


_X13_EXECUTABLE_FILE = {
    "Windows": "x13as.exe",
    "Linux": "x13asunix",
    "Darwin": "x13asmac",
}[_pf.system()]


_X13_EXECUTABLE_PATH = _os.path.join(_EXECUTABLES_PATH, _X13_EXECUTABLE_FILE, )


