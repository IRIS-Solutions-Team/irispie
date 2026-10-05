# Code by OGResearch
"""
Copy of patch_irispie_preparser from
modules/codes/python/general/utils/irispie_compat.py, so that this folder
runs without ogi.

irispie 0.79.6 (and 0.80.1) has a defect in its model preparser: an `!if`
block without an `!else` picks up the `!else` of a later block, and fails with
"Misplaced preparsing directive !end" whenever the condition is true.
gpm.model has 81 `!if` and 12 `!else`, and 38 blocks are affected, so it
does not parse on a stock irispie. The patch bounds the `!else` search by
the matching `!end`. It probes first and does nothing on a fixed irispie.
"""

_PREPARSER_ELSE_BUG_SOURCE = """
!transition-variables
  a
!if flag !then
  b
!end
!if other !then
  c
!else
  d
!end
!transition-equations
  a = 1;
"""


def _has_preparser_else_bug(preparser) -> bool:
    """
    Run the minimal reproduction and report whether the defect is present.

    Returns True only on the specific "Misplaced preparsing directive !end"
    failure; any other exception is re-raised, because it means the probe
    itself no longer matches the installed irispie and the result would be
    meaningless.
    """
    try:
        preparser.from_string(
            _PREPARSER_ELSE_BUG_SOURCE,
            context={"flag": True, "other": True},
        )
    except Exception as exc:
        if "Misplaced preparsing directive" in str(exc):
            return True
        raise
    return False


def patch_irispie_preparser() -> None:
    """
    Bound the "!else" search of the model preparser by the matching "!end".

    Does nothing when the installed irispie resolves the reproduction above
    correctly, or when its preparser no longer exposes the two internals the
    patch needs.

    Note: this function has no Matlab counterpart.

    TODO: remove once irispie fixes "_find_matching_else" upstream.
    Reported to the irispie authors on 2026-09-22; present in 0.79.6 and
    unchanged in 0.80.1 ("preparser.py" is byte-identical between the two).
    RECHECK ON EVERY IRISPIE UPGRADE: run "_has_preparser_else_bug" against
    the new version, and when it returns False delete this function, its
    reproduction source and the call in "general/__init__.py".
    Until then the patch is inert on a fixed irispie anyway -- it probes
    first and returns without touching anything -- so an upgrade is safe
    without this cleanup.
    """

    try:
        from irispie.parsers import preparser as _preparser
    except ImportError:
        return

    required = ("_find_matching_else", "_find_matching_end", "_cumulate_level")
    if not all(hasattr(_preparser, name) for name in required):
        return

    if getattr(_preparser, "_ogi_else_patch_applied", False):
        return

    if not _has_preparser_else_bug(_preparser):
        return

    else_class = _preparser._Else

    def _find_matching_else(sequence):
        """Find the !else of this block, i.e. before the matching !end."""
        cum_level = _preparser._cumulate_level(sequence)
        index_end = cum_level.index(0)
        return next(
            (i for i, level in enumerate(cum_level[:index_end])
             if level == 1 and isinstance(sequence[i], else_class)),
            None,
        )

    _preparser._find_matching_else = _find_matching_else
    _preparser._ogi_else_patch_applied = True

    if _has_preparser_else_bug(_preparser):
        raise RuntimeError(
            "patch_irispie_preparser did not fix the !if/!else matching "
            "defect; the preparser internals have changed and the patch "
            "needs to be revisited.")
