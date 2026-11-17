"""Scalp derivations. Every derivation is a matrix M acting on referential data; rows of
average-reference and Laplacian matrices sum to zero, so the recording reference cancels exactly."""
from __future__ import annotations

import numpy as np

OLD2NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
DS4752_SCALP = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]  # A1/A2 excluded


def std1020_positions(names):
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return np.array([pos[OLD2NEW.get(n, n)] for n in names]) * 1000.0      # mm


def avgref_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n


def hjorth_matrix(pos, k=4):
    """Nearest-neighbour (Hjorth) Laplacian: x_i - mean of its k nearest channels."""
    n = len(pos)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    M = np.eye(n)
    for i in range(n):
        nb = np.argsort(D[i])[1:k + 1]
        M[i, nb] = -1.0 / k
    return M


def pair_distances(pos):
    a, b = np.triu_indices(len(pos), 1)
    return np.linalg.norm(pos[a] - pos[b], axis=-1)


# ---------------------------------------------------------------- real spherical harmonics
def sph_basis(pos, nmax):
    """Real spherical-harmonic design matrix on electrode directions. Returns (Y [n_ch, n_coef], degree [n_coef])."""
    from scipy.special import sph_harm_y
    p = pos - pos.mean(0)
    r = np.linalg.norm(p, axis=1)
    theta = np.arccos(np.clip(p[:, 2] / r, -1, 1))       # polar
    phi = np.arctan2(p[:, 1], p[:, 0])
    cols, deg = [], []
    for n in range(0, nmax + 1):
        for m in range(-n, n + 1):
            Y = sph_harm_y(n, abs(m), theta, phi)
            if m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            elif m == 0:
                cols.append(Y.real)
            else:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
            deg.append(n)
    return np.stack(cols, 1), np.array(deg)
