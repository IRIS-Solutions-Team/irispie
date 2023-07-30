"""
"""


#[
from __future__ import annotations

import numpy as np_
import scipy as sp_

from ..fords import (systems as sy_, )
#]


def solve_steady_linear_flat(
    sys: sy_.System,
    /,
) -> tuple[np_.ndarray, np_.ndarray, np_.ndarray, np_.ndarray]:
    """
    """
    #[
    """
    """
    pinv = np_.linalg.pinv
    lstsq = sp_.linalg.lstsq
    vstack = np_.vstack
    hstack = np_.hstack
    A, B, C, F, G, H = sys.A, sys.B, sys.C, sys.F, sys.G, sys.H
    #
    # A @ Xi + B @ Xi{-1} + C = 0
    # F @ Y + G @ Xi + H = 0
    #
    # Xi = -pinv(A + B) @ C
    Xi, *_ = lstsq(-(A + B), C)
    dXi = np_.zeros(Xi.shape)
    #
    # Y = -pinv(F) @ (G @ Xi + H)
    Y, *_ = lstsq(-F, G @ Xi + H)
    dY = np_.zeros(Y.shape)
    #
    return Xi, Y, dXi, dY
    #]


def solve_steady_linear_nonflat(
    sys: sy_.System,
    /,
) -> tuple[np_.ndarray, np_.ndarray, np_.ndarray, np_.ndarray]:
    """
    """
    #[
    """
    """
    # pinv = np_.linalg.pinv
    lstsq = np_.linalg.lstsq
    vstack = np_.vstack
    hstack = np_.hstack
    A, B, C, F, G, H = sys.A, sys.B, sys.C, sys.F, sys.G, sys.H
    num_y = F.shape[0]
    k = 1
    #
    # A @ Xi + B @ Xi{-1} + C = 0:
    # -->
    # A @ Xi + B @ (Xi - dXi) + C = 0
    # A @ (Xi + k*dXi) + B @ (Xi + (k-1)*dXi) + C = 0
    #
    AB = vstack((
        hstack(( A + B, 0*A + (0-1)*B )),
        hstack(( A + B, k*A + (k-1)*B )),
    ))
    CC = vstack((
        C,
        C,
    ))
    # Xi_dXi = -pinv(AB) @ CC
    Xi_dXi, *_ = lstsq(-AB, CC, rcond=None)
    #
    # F @ Y + G @ Xi + H = 0:
    # -->
    # F @ Y + G @ Xi + H = 0
    # F @ (Y + k*dY) + G @ (Xi + k*dXi) + H = 0
    #
    FF = vstack((
        hstack(( F, 0*F )),
        hstack(( F, k*F )),
    ))
    GG = vstack((
        hstack(( G, 0*G )),
        hstack(( G, k*G )),
    ))
    HH = vstack((
        H,
        H,
    ))
    # Y_dY = -pinv(FF) @ (GG @ Xi_dXi + HH)
    Y_dY, *_ = lstsq(-FF, GG @ Xi_dXi + HH, rcond=None)
    #
    # Separate levels and changes
    #
    num_xi = A.shape[1]
    num_y = F.shape[1]
    Xi, dXi = (
        Xi_dXi[0:num_xi, ...],
        Xi_dXi[num_xi:, ...],
    )
    Y, dY = (
        Y_dY[0:num_y, ...],
        Y_dY[num_y:, ...]
    )
    #
    return Xi, Y, dXi, dY
    #]


