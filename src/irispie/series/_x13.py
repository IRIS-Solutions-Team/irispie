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
import copy as _co
import glob as _gl
import documark as _dm

from .. import executables as _executables
from .. import wrongdoings as _wrongdoings
from .. import has_variants as _has_variants
from ..dates import (Period, Span, Frequency, )

from ._functionalize import FUNC_STRING

if TYPE_CHECKING:
    from typing import (Any, )
    from types import (EllipsisType, )
    from ..series import (Series, )
#]


__all__ = []


Mode = Literal["mult", "add", "pseudoadd", "logadd", ]


class Inlay:
    """
    """
    #[

    @_dm.reference(category="filtering", )
    def x13(
        self,
        *,
        span: Span | EllipsisType = ...,
        when_error: _wrongdoings.HOW = "warning",
        clean_up: bool = True,
        output: str = "seasonally_adjusted",
        #
        return_info: bool = False,
        unpack_singleton: bool = True,
        #
        specs_template: dict[str, Any] | None = None,
        mode: Literal["mult", "add", "pseudoadd", "logadd", ] | None = None,
        allow_missing: bool = False,
        add_to_specs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        r"""
................................................................................

==X13-ARIMA-TRAMO-SEATS seasonal adjustment procedure==


### Function for creating new Series objects ###


```
new = irispie.x13(
    self,

    span=None,
    output="seasonally_adjusted",
    mode=None,
    when_error="warning",
    clean_up=True,

    specs_template=None,
    add_to_specs=None,
    allow_missing=False,
    mode=None,

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


### Class method for changing existing time `Series` objects in-place ###


```
self.x13(
    self,

    span=None,
    output="seasonally_adjusted",
    mode=None,
    when_error="warning",
    clean_up=True,

    specs_template=None,
    add_to_specs=None,
    allow_missing=False,
    mode=None,

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
    available at the moment:

    | Output                  | X13 table | Description
    |-------------------------|-----------|-------------
    | `"seasonal"`            | `d10`     | Seasonal factors
    | `"seasonally_adjusted"` | `d11`     | Seasonally adjusted series
    | `"trend_cycle"`         | `d12`     | Trend-cycle component
    | `"irregular"`           | `d13`     | Irregular component
    | `"seasonal_and_td"`     | `d16`     | Combined seasonal and trading day factors
    | `"holiday_and_td"`      | `d18`     | Combined holiday and trading day factors

???+ input "specs_template"
    A dictionary with a specs template for the X13 run; if `None`, a default
    specs template is used (see below for the structure of the default template).

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

???+ input "allow_missing"
    If `True`, allow missing values in the input time series and automatically
    add an empty `automdl` spec if no ARIMA model is specified.

???+ input "add_to_specs"
    A dictionary with additional settings to be added to the `specs_template` (or
    the default templated).

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
    | `success` | True if the X13 run was successful
    | `specs_template` | The specs template used for the X13 run
    | `mode` | The mode used for the X13 run
    | `spc` | The spc file from the X13 run
    | `log` | The log file from the X13 run
    | `out` | The output file from the X13 run
    | `err` | The error file from the X13 run
    | `*` | Any other output file written by X13


### Details ###


???+ abstract "Default SPC template structure"

    The default specs template is a dictionary equivalent to the following SPC
    file:

    ```
    series{
        start=$(series_start)
        data=(
    $(series_data)
        )
        period=$(series_period)
        decimals=5
        precision=5
    }

    transform{
        function=$(transform_function)
    }

    x11{
        mode=$(x11_mode)
        save=$(x11_save)
    }

    ```

................................................................................
        """
        span = self.resolve_periods(span)
        base_start, base_end = span[0], span[-1]
        self.clip(base_start, base_end, )
        #
        mode, transform_function, flip_sign = _resolve_mode(self, mode, )
        basic_settings = _prepare_basic_settings(
            base_start, output,
            mode=mode,
            transform_function=transform_function,
        )
        specs_template = _get_specs_template(specs_template, )
        _update_specs_template_with_extra_settings(specs_template, add_to_specs, allow_missing, )
        specs_template_string = _print_specs_template(specs_template, )
        specs_template_string = _insert_basic_settings_into_specs_template_string(specs_template_string, basic_settings, )
        #
        new_data = []
        out_info = []
        for variant_data in self.data.T:
            info_v = {
                "specs_template": specs_template,
                "mode": mode,
                "transform_function": transform_function,
                "flip_sign": flip_sign,
            }
            if flip_sign:
                variant_data = -variant_data
            new_data_v = _x13_data(
                specs_template_string,
                base_start,
                variant_data,
                basic_settings["x11_save"],
                info=info_v,
                clean_up=clean_up,
            )
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


attributes = (n for n in dir(Inlay) if not n.startswith("_"))
for n in attributes:
    code = FUNC_STRING.format(n=n, )
    exec(code, globals(), locals(), )
    __all__.append(n)


def _x13_data(
    specs: str,
    base_start: Period,
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
    year, segment = start_date.to_year_segment()
    specs = specs.replace("$(series_start)", _X13_DATE_FORMAT_RESOLUTION[start_date.frequency].format(year=year, segment=segment, ), )
    specs = specs.replace("$(series_data)", _print_series_data(variant_data, ), )
    specs_file_name_without_ext = _write_specs_to_file(specs, )
    system_output = _execute(specs_file_name_without_ext, )
    raw_output_data, update_info = _collect_outputs(specs_file_name_without_ext, system_output, x11_save, )
    info.update({"specs": specs, })
    info.update(update_info, )
    if raw_output_data is not None:
        new_data[output_slice] = raw_output_data
    if clean_up:
        _clean_up(specs_file_name_without_ext, )
    return new_data
    #]


def _print_series_data(
    variant_data: _np.ndarray,
    /,
) -> str:
    """
    """
    _INDENT = " " * 8
    _NAN_REPLACEMENT = "-99999.0"
    data_str = "\n".join(
        _INDENT + (f"{v:g}" if not _np.isnan(v) else _NAN_REPLACEMENT)
        for v in variant_data
    )
    return data_str


def _execute(
    specs_file_name_without_ext: str,
    /,
) -> _sp.CompletedProcess:
    """
    """
    #[
    return _sp.run(
        [_X13_EXECUTABLE_PATH, specs_file_name_without_ext, ],
        stdout=_sp.PIPE,
        check=True,
    )
    #]


def _collect_outputs(
    specs_file_name_without_ext: str,
    output: _sp.CompletedProcess,
    out_table: str,
    /,
) -> tuple[_np.ndarray | None, dict[str, str]]:
    """
    """
    #[
    info = _read_out_files(specs_file_name_without_ext, )
    try:
        info[out_table] = str(info[out_table])
        out_data = _read_output_data(info[out_table], )
    except:
        out_data = None
        info[out_table] = None
    info["success"] = (
        output.returncode == 0
        and "ERROR" not in str(output.stdout)
        and out_data is not None
    )
    return out_data, info
    #]


def _read_output_data(
    out_table: str,
) -> tuple[tuple[float, ...], str]:
    """
    """
    return tuple(float(i) for i in out_table.split()[5::2])


def _read_out_files(
    specs_file_name_without_ext: str,
) ->  dict[str, str]:
    """
    """
    #[
    info = dict()
    for file_name in _gl.glob(specs_file_name_without_ext + ".*"):
        ext = file_name.split(".")[-1]
        encoding = None if ext != "err" else "latin-1"
        with open(file_name, "rt", encoding=encoding, ) as fid:
            info[ext] = fid.read()
    return info
    #]


def _write_specs_to_file(
    specs: str,
    /,
) -> str:
    """
    """
    with _tf.NamedTemporaryFile(dir=".", mode="wt", suffix=".spc", delete=False, ) as fid:
        specs_file_name = fid.name
        fid.write(specs)
    return specs_file_name.removesuffix(".spc")


def _remove_leading_trailing_nans(
    base_start: Period,
    variant_data: _np.ndarray,
    /,
) -> tuple[Period, _np.ndarray, slice]:
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


def _prepare_basic_settings(
    base_start: Period,
    output: str,
    mode: Mode,
    transform_function: str = "none",
) ->  dict[str, str]:
    """
    """
    #[
    return {
        "series_period": str(base_start.frequency.value),
        "x11_mode": str(mode),
        "x11_save": _X11_OUTPUT_RESOLUTION.get(output, output),
        "transform_function": str(transform_function),
    }
    #]


def _get_specs_template(
    specs_template: dict[str, Any] | None,
) -> dict[str, Any]:
    """Get specs template either from a custom dict or a default dict"""
    return _co.deepcopy(specs_template or _DEFAULT_TEMPLATE)


def _update_specs_template_with_extra_settings(
    specs_template: dict[str, Any] | None,
    add_to_specs: dict[str, Any] | None,
    allow_missing: bool,
) -> None:
    """Update specs template with extra custom settings"""
    #[
    if add_to_specs:
        for k, v in add_to_specs.items():
            if k in specs_template:
                specs_template[k].update(v, )
            else:
                specs_template[k] = v
    _add_arima_model_if_needed(specs_template, allow_missing, )
    #]


def _add_arima_model_if_needed(
    specs_template: dict[str, Any],
    allow_missing: bool,
) -> None:
    """Add arima model if not present and in-sample missing values are allowed"""
    #[
    if not allow_missing:
        return
    has_arima = (
        ("automdl" in specs_template and specs_template["automdl"] is not False)
        or ("arima" in specs_template and specs_template["arima"] is not False)
    )
    if not has_arima:
        specs_template["automdl"] = {}
    #]


def _print_specs_template(
    specs_template: dict[str, Any],
) -> str:
    """Print the specs template dict to a string"""
    #[
    specs_template_string = "\n"
    for k, v in specs_template.items():
        if v is False:
            continue
        add = "\n\n" + k.lower() + "{"
        if isinstance(v, dict) and v:
            for kk, vv in v.items():
                add += f"\n    {kk.lower()}={vv}"
        add += "\n}"
        specs_template_string += add
    specs_template_string += "\n"
    return specs_template_string
    #]


def _insert_basic_settings_into_specs_template_string(
    specs_template_string: str,
    settings: dict[str, str],
) -> str:
    """
    Replace placeholders with custom setting
    """
    #[
    for k, v in settings.items():
        specs_template_string = specs_template_string.replace(f"$({k})", v)
    return specs_template_string
    #]


def _clean_up(
    specs_file_name_without_ext: str,
    /,
) -> None:
    """
    """
    #[
    for f in _gl.glob(specs_file_name_without_ext + ".*"):
        _os.remove(f, )
    #]


_EXECUTABLES_PATH = _os.path.dirname(_executables.__file__)

_DEFAULT_SERIES_TEMPLATE = {
    "start": "$(series_start)",
    "data": "(\n$(series_data)\n    )",
    "period": "$(series_period)",
    "decimals": 5,
    "precision": 5,
}

_DEFAULT_TRANSFORM_TEMPLATE = {
    "function": "$(transform_function)",
}

_DEFAULT_X11_TEMPLATE = {
    "mode": "$(x11_mode)",
    "save": "$(x11_save)",
}

_DEFAULT_TEMPLATE = {
    "series": _DEFAULT_SERIES_TEMPLATE,
    "transform": _DEFAULT_TRANSFORM_TEMPLATE,
    "x11": _DEFAULT_X11_TEMPLATE,
}

_X11_OUTPUT_RESOLUTION = {
    "sf": "d10",
    "seasonal": "d10",
    "seasonal_factors": "d10",
    "sa": "d11",
    "seasonally_adjusted": "d11",
    "tc": "d12",
    "trend_cycle": "d12",
    "irr": "d13",
    "irregular": "d13",
    "seasonal_and_td": "d16",
    "holiday_and_td": "d18",
}

_X13_DATE_FORMAT_RESOLUTION = {
    Frequency.MONTHLY: "{year:04d}.{segment:02d}",
    Frequency.QUARTERLY: "{year:04d}.{segment:1d}",
}

_X13_EXECUTABLE_FILE = {
    "Windows": "x13aswin.exe",
    "Linux": "x13asunix",
    "Darwin": "x13asmac",
}[_pf.system()]


_X13_EXECUTABLE_PATH = _os.path.join(_EXECUTABLES_PATH, _X13_EXECUTABLE_FILE, )


