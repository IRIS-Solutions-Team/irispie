"""
"""


CURRENT_PORTABLE_FORMAT = "0.1.0"

LEGACY_PORTABLE_FORMATS = set()

SUPPORTED_PORTABLE_FORMATS = {CURRENT_PORTABLE_FORMAT, } | LEGACY_PORTABLE_FORMATS


def validate_portable_format(portable_format: str, /, ) -> None:
    if portable_format in SUPPORTED_PORTABLE_FORMATS:
        return
    raise _wrongdoings.IrisPieCritical(f"Portable format {portable_format} not supported")

