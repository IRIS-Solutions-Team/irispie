"""
Differentiation and invariation rules
"""

#[
from __future__ import annotations
#]

def add_diff(self, other):
    """
    Differentiate self(x)+other(x) or self(x)+other
    """
    if hasattr(other, "_is_atom"):
        new_value = self.value + other.value
        new_diff = self.diff + other.diff
    else:
        new_value = self.value + other
        new_diff = self.diff
    return type(self).no_context(new_value, new_diff, False)

def add_invar(self, other):
    """
    Invariance of self(x)+other(x) or self(x)+other
    """
    if hasattr(other, "_is_atom"):
        new_diff = self._diff | other._diff
        new_invariant = self._invariant & other._invariant
    else:
        new_diff = self._diff
        new_invariant = self._invariant
    return Atom.no_context(new_diff, new_invariant, False)



def sub_diff(self, other):
    if hasattr(other, "_is_atom"):
        new_value = self.value - other.value
        new_diff = self.diff - other.diff
    else:
        new_value = self.value - other
        new_diff = self.diff
    new_logly = False
    return type(self).no_context(new_value, new_diff, False)

sub_invar = add_invar


