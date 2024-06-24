"""
Interface to X13-ARIMA-TRAMO-SEATS
"""


#[
from __future__ import annotations

from typing import (Literal, TYPE_CHECKING, )
import os as _os
import numpy as _np
import subprocess as _sp
import platform as _pf
import tempfile as _tf
import glob as _gl

from .. import executables as _executables
from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants
from .. import dates as _dates
from .. import pages as _pages

from . import _functionalize

if TYPE_CHECKING:
    from typing import (Any, )
    from types import (EllipsisType, )
    from ..series import (Series, )
#]


__all__ = ()


Mode = Literal["mult", "add", "pseudoadd", "logadd", ]


class Inlay:
    """
    """
    #[

    @_pages.reference(category="filtering", )
    def x13(
        self,
        *,
        span: _dates.Span | EllipsisType = ...,
        when_error: _wrongdoings.HOW = "warning",
        clean_up: bool = True,
        output: str = "sa",
        mode: Literal["mult", "add", "pseudoadd", "logadd", ] | None = None,
        #
        return_info: bool = False,
        unpack_singleton: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
................................................................................

==X13-ARIMA-TRAMO-SEATS seasonal adjustment procedure==


### Function for creating new Series objects ###


```
new = irispie.x13(
    self,
    /,
    span=None,
    output="sa",
    mode=None,
    when_error="warning",
    clean_up=True,
    unpack_singleton=True,
    return_info=False,
)
```

```
new, info = irispie.x13(
    ...,
    return_info=True,
    ...,
)
```


### Class methods for changing existing Series objects in-place ###


```
self.x13(
    span=None,
    output="sa",
    mode=None,
    when_error="warning",
    clean_up=True,
    unpack_singleton=True,
    return_info=False,
)
```

```
info = self.x13(
    ...,
    return_info=True,
    ...,
)
```


### Input arguments ###


???+ input "self"
    A time `Series` object whose data will be run through the
    X13-ARIMA-TRAMO-SEATS procedure.

???+ input "span"
    A time span be specified as a `Span` object. If `span=None` or `span=...`,
    the time span goes from the first observed period to the last observed
    period in the input time series.

???+ input "output"
    The type of output to be returned by X13. The following options are
    available:

    | Output    | Description
    |-----------|-------------
    | `"sf"`    | Seasonal factors
    | `"sa"`    | Seasonally adjusted series
    | `"tc"`    | Trend-cycle
    | `"irr"`   | Irregular component

???+ input "mode"
    The mode to be used for the X13 run. The following options are available (see the
    [X13 documentation](https://www.census.gov/srd/www/x13as/)):

    | Mode          | Description
    |---------------|-------------
    | `None`        | Automatically selected
    | `"mult"`      | Multiplicative
    | `"add"`       | Additive
    | `"pseudoadd"` | Pseudo-additive
    | `"logadd"`    | Log-additive

    If `mode=None`, the mode is automatically selected based on the data. If the data is
    strictly positive or strictly negative, the multiplicative mode is used, otherwise
    the additive mode is used.

???+ input "when_error"
    The action to be taken when an error occurs. The following options are
    available:

    | Action      | Description
    |-------------|-------------
    | `"warning"` | Issue a warning
    | `"error"`   | Raise an error

???+ input "unpack_singleton"
    If `True`, unpack `info` into a plain dictionary for models with a
    single variant.

???+ input "return_info"
    If `True`, return a dictionary with information about the X13 run as another
    output argument.


### Returns ###


???+ returns "self"
    The `Series` object with the output data.

???+ returns "new"
    A new `Series` object with the output data.

???+ returns "info"
    (Only returned if `return_info=True` which is not the default behavior)
    Dictionary with information about the X13 run; `info` contains the
    following items:

    | Key | Description
    |-----|-------------
    | `mode` | The mode used for the X13 run
    | `log` | The log file from the X13 run
    | `out` | The output file from the X13 run
    | `err` | The error file from the X13 run
    | `success` | A boolean indicating whether the X13 run was successful


### Details ###


................................................................................
        """
        span = tuple(self._resolve_dates(span))
        base_start, base_end = span[0], span[-1]
        self.clip(base_start, base_end, )
        #
        mode, transform_function, flip_sign = _resolve_mode(self, mode, )
        settings = _create_settings(
            base_start, output,
            mode=mode,
            transform_function=transform_function,
            **kwargs,
        )
        spc = _create_spc_file(_TEMPLATE_SPC, settings, )
        #
        new_data = []
        out_info = []
        for variant_data in self.data.T:
            info_v = {
                "mode": mode,
                "transform_function": transform_function,
                "flip_sign": flip_sign,
            }
            #
            if flip_sign:
                variant_data = -variant_data
            #
            new_data_v = _x13_data(
                spc, base_start, variant_data, settings["x11_save"],
                info=info_v,
                clean_up=clean_up,
            )
            #
            if flip_sign:
                new_data_v = -new_data_v
            new_data.append(new_data_v.reshape(-1, 1, ))
            out_info.append(info_v)
            #
        if not all(i["success"] for i in out_info):
            message = "X13 failed to produce a result for at least one variant."
            _wrongdoings.raise_as(when_error, message, )
        #
        new_data = _np.hstack(new_data, )
        self._replace_start_and_values(base_start, new_data, )
        #
        if return_info:
            out_info = _has_variants.unpack_singleton(
                out_info, self.is_singleton,
                unpack_singleton=unpack_singleton,
            )
            return out_info
        else:
            return
    #]


for n in ("x13", ):
    exec(_functionalize.FUNC_STRING.format(n=n, ), globals(), locals(), )
    __all__ += (n, )


def _x13_data(
    spc: str,
    base_start: _dates.Date,
    base_variant_data: _np.ndarray,
    x11_save: str,
    /,
    *,
    info: dict[str, Any],
    clean_up: bool = True,
) -> tuple[_np.ndarray, dict[str, Any]]:
    """
    """
    #[
    new_data = _np.full(base_variant_data.shape, _np.nan, dtype=float, )
    start_date, variant_data, output_slice = _remove_leading_trailing_nans(base_start, base_variant_data, )
    year, period = start_date.to_year_period()
    spc = spc.replace("$(series_start)", _X13_DATE_FORMAT_RESOLUTION[start_date.frequency].format(year=year, period=period, ), )
    spc = spc.replace("$(series_data)", _print_series_data(variant_data, ), )
    spc_file_name_without_ext = _write_spc_to_file(spc, )
    system_output = _execute(spc_file_name_without_ext, )
    raw_output_data, update_info = _collect_outputs(spc_file_name_without_ext, system_output, x11_save, )
    info.update(update_info, )
    if raw_output_data is not None:
        new_data[output_slice] = raw_output_data
    if clean_up:
        _clean_up(spc_file_name_without_ext, )
    return new_data
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
    info = _read_out_info(spc_file_name_without_ext, )
    info["success"] = output.returncode == 0 and "ERROR" not in str(output.stdout)
    if info["success"]:
        out_data, info[x11_save] = _read_output_data(spc_file_name_without_ext, x11_save, info, )
    else:
        out_data = None
        info[x11_save] = None
    return out_data, info
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
    out_data = tuple(float(i) for i in table.split()[5::2])
    return out_data, table
    #]


def _read_out_info(
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
    base_start: _dates.Date,
    variant_data: _np.ndarray,
    /,
) -> tuple[_dates.Date, _np.ndarray, slice]:
    """
    """
    #[
    start_date = base_start.copy()
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


_TRANFORM_FUNCTION_DISPATCH = {
    "mult": "log",
}


def _resolve_mode(
    self: Series,
    mode: Mode | None,
    transform_function: str | None = None,
) -> tuple[str, str, bool]:
    """
    """
    #[
    flip_sign = False
    if mode is None:
        is_sign_strict = _np.all(self.data > 0) or _np.all(self.data < 0)
        mode = "mult" if is_sign_strict else "add"
    if transform_function is None:
        transform_function = _TRANFORM_FUNCTION_DISPATCH.get(mode, "none", )
    flip_sign = _np.all(self.data < 0)
    return mode, transform_function, flip_sign
    #]


def _create_settings(
    base_start: _dates.Date,
    output: str,
    /,
    *,
    mode: Mode,
    transform_function: str = "none",
) ->  dict[str, str]:
    """
    """
    #[
    settings = {
        "series_period": str(base_start.frequency.value),
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
    "Windows": "x13aswin.exe",
    "Linux": "x13asunix",
    "Darwin": "x13asmac",
}[_pf.system()]


_X13_EXECUTABLE_PATH = _os.path.join(_EXECUTABLES_PATH, _X13_EXECUTABLE_FILE, )


