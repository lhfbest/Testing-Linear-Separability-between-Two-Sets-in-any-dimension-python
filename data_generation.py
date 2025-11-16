from __future__ import annotations

import numpy as np


def generate_ls_data(n: int, m: int, shift_value: float = 500.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate linearly separable data similar to MATLAB Generate_LS_data.

    Returns array with shape (n+1, m_total): first row labels (0 for A, 1 for B),
    below are coordinates, points as columns.
    """
    if rng is None:
        rng = np.random.default_rng()

    points = rng.integers(-1000, 1000, size=(n, m)).astype(float)
    normal_vector = rng.standard_normal(size=(n,))
    normal_vector /= np.linalg.norm(normal_vector) + 1e-15

    distances = points.T @ normal_vector
    sorted_indices = np.argsort(distances)
    num_A = int(np.floor(m / 4))
    A_idx = sorted_indices[:num_A]
    B_idx = sorted_indices[num_A:]

    A = points[:, A_idx]
    B = points[:, B_idx]

    A = A - normal_vector[:, None] * shift_value
    B = B + normal_vector[:, None] * shift_value

    labels_A = np.zeros((1, A.shape[1]))
    labels_B = np.ones((1, B.shape[1]))
    merged = np.concatenate([np.hstack([labels_A, labels_B]), np.hstack([A, B])], axis=0)
    return merged


def generate_nls_data(n: int, m: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate non-linearly separable data similar to MATLAB Generate_NLS_data.
    Returns array with shape (n+1, m_total): first row labels.
    """
    if rng is None:
        rng = np.random.default_rng()

    m_effective = m - 4
    num_A = int(np.floor(m_effective / 3))
    num_B = m_effective - num_A

    points_A = rng.random(size=(n, num_A)) * 2000.0 - 1000.0

    indices = rng.integers(0, num_A, size=(num_B,))
    noise = rng.standard_normal(size=(n, num_B))
    points_B = points_A[:, indices] + noise

    point1_A = np.zeros((n, 1))
    point2_A = np.zeros((n, 1))
    if n >= 2:
        point2_A[0, 0] = 1.0
        point2_A[1, 0] = 1.0
    points_A = np.hstack([point1_A, point2_A, points_A])

    point1_B = np.zeros((n, 1))
    point2_B = np.zeros((n, 1))
    if n >= 2:
        point1_B[1, 0] = 1.0
        point2_B[0, 0] = 1.0
    points_B = np.hstack([point1_B, point2_B, points_B])

    labels = np.hstack([np.zeros((1, points_A.shape[1])), np.ones((1, points_B.shape[1]))])
    data = np.vstack([labels, np.hstack([points_A, points_B])])
    return data


__all__ = ["generate_ls_data", "generate_nls_data"]


