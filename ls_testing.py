import time
from typing import Tuple

import numpy as np


def _nullspace(matrix: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Compute an orthonormal basis for the nullspace of a matrix using SVD.

    Returns columns forming a basis. If the nullspace is 0-dimensional, returns
    an array of shape (n, 0).
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = (s > tol * s.max() if s.size > 0 else np.array([])).sum()
    nullspace_basis = vh[rank:].T
    return nullspace_basis


def _project_to_affine_hull(support_points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the affine hull of the support points.

    support_points: shape (n, m)
    vector: shape (n,)
    """
    n, m = support_points.shape
    pivot = support_points[:, -1]
    w = support_points - pivot[:, None]
    if m == 1:
        # Affine hull is just the point itself
        return pivot.copy()
    # Project onto the linear span of columns of w, then shift back through pivot
    w_pinv = np.linalg.pinv(w)
    projection = pivot + w @ (w_pinv @ (vector - pivot))
    return projection


def _barycentric_coords(support_points: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of vector w.r.t. support_points.
    Solve [1; S] * lambda = [1; vector].
    """
    n, m = support_points.shape
    M = np.vstack([np.ones((1, m)), support_points])
    rhs = np.concatenate(([1.0], vector))
    # Least squares in case of slight numerical issues
    lambdas, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return lambdas


def _affinely_independent_after_append(support_points: np.ndarray, p: np.ndarray, eps: float) -> bool:
    n, m = support_points.shape
    if m + 1 > n + 1:
        return False
    M = np.vstack([np.ones((1, m)), support_points])
    M_aug = np.hstack([M, np.concatenate(([1.0], p))[:, None]])
    rank_before = np.linalg.matrix_rank(M, eps)
    rank_after = np.linalg.matrix_rank(M_aug, eps)
    return rank_after == rank_before + 1


def _large_update_s(support_points: np.ndarray, p: np.ndarray, x: np.ndarray, eps: float,
                    trace_fn=None) -> np.ndarray:
    """
    Update support set S to ensure {S, p} is affinely independent.
    Implements a minimum-ratio rule analogous to the MATLAB version when needed.
    """
    n, m = support_points.shape
    if _affinely_independent_after_append(support_points, p, eps):
        if trace_fn is not None:
            trace_fn({
                "event": "updateS_affine_independent",
                "m_before": m,
            })
        return support_points

    # Build matrix M = [1; S]
    M = np.vstack([np.ones((1, m)), support_points])
    # Targets: columns [-1; -p] and [1; x]
    target = np.column_stack([np.concatenate(([-1.0], -p)), np.concatenate(([1.0], x))])
    # Solve M Z = target
    Z, *_ = np.linalg.lstsq(M, target, rcond=None)
    # Minimum ratio rule
    denominators = Z[:, 0]
    numerators = -Z[:, 1]
    min_ratio = np.inf
    index_to_remove = None
    for i in range(m):
        if denominators[i] < 0:
            ratio = numerators[i] / denominators[i]
            if ratio < min_ratio:
                min_ratio = ratio
                index_to_remove = i
    if index_to_remove is None:
        # Fallback: remove the first column if numerical issues occur
        index_to_remove = 0
    if trace_fn is not None:
        trace_fn({
            "event": "updateS_remove",
            "m_before": m,
            "remove_index": int(index_to_remove),
            "min_ratio": float(min_ratio) if np.isfinite(min_ratio) else None,
            "den_first3": denominators[:3].tolist() if m >= 1 else [],
            "num_first3": numerators[:3].tolist() if m >= 1 else [],
        })
    mask = np.ones(m, dtype=bool)
    mask[index_to_remove] = False
    return support_points[:, mask]


def _qr_line_search(x: np.ndarray, S: np.ndarray, p: np.ndarray, eps: float,
                    trace_fn=None) -> Tuple[np.ndarray, int]:
    """
    QR-based line search mirroring the MATLAB implementation.
    """
    n, m = S.shape
    pm = S[:, -1]

    # Build S_bar = [S with last col = p] - pm * 1^T
    S_tilde = S.copy()
    S_tilde[:, -1] = p
    ones_row = np.ones((1, m))
    S_bar = S_tilde - pm[:, None] @ ones_row

    # Full QR of S_bar
    Q2, R2 = np.linalg.qr(S_bar, mode='complete')

    # Build A with first m rows = R2', and lower-right identity
    A = np.zeros((n, n), dtype=S.dtype)
    A[:m, :] = R2.T[:m, :]
    if n > m:
        A[m:, m:] = np.eye(n - m, dtype=S.dtype)
    # Solve for g, then direction d = Q2 * g
    ee = np.zeros((n,), dtype=S.dtype)
    ee[m - 1] = 1.0
    # Use solve or least squares depending on conditioning
    try:
        g = np.linalg.solve(A, ee)
    except np.linalg.LinAlgError:
        g, *_ = np.linalg.lstsq(A, ee, rcond=None)
    d = Q2 @ g

    if trace_fn is not None:
        _norm_d = float(np.linalg.norm(d))
        trace_fn({
            "event": "line_dir",
            "m": int(m),
            "||d||": _norm_d,
            "norm_d": _norm_d,
            "dot_pmmp_d": float((pm - p) @ d),
        })

    # Alpha for bisector with p
    denom_bis = (pm - p) @ d
    alpha_bis = np.inf if abs(denom_bis) < 1e-15 else ((pm - p) @ (0.5 * (p + pm) - x)) / denom_bis

    # Q3 from deleting last column of S_bar
    if m > 1:
        S_bar_mdel = S_bar[:, :m - 1]
        Q3_del, _ = np.linalg.qr(S_bar_mdel, mode='complete')
        V = Q3_del[:, :m - 1] @ Q3_del[:, :m - 1].T
    else:
        V = np.zeros((n, n), dtype=S.dtype)

    Vpm = V @ pm
    xx = V @ x - Vpm + pm
    pp = V @ p - Vpm + pm

    # Affine system for pi, omega: M_aug = [1; S]
    M_aug = np.vstack([np.ones((1, m), dtype=S.dtype), S])
    B = np.column_stack([np.concatenate(([1.0], xx)), np.concatenate(([1.0], pp))])
    # Solve M_aug * Z = B
    Z, *_ = np.linalg.lstsq(M_aug, B, rcond=None)
    pi = Z[:, 0]
    omega = Z[:, 1]

    # Case 1: any j with pi(j)=0 and omega(j)=0?
    omTmp = omega.copy()
    omTmp[(omTmp < -eps) | (omTmp > eps)] = np.inf
    piTmp = pi.copy()
    piTmp[(piTmp < -eps) | (piTmp > eps)] = np.inf
    minSum = np.min(piTmp + omTmp)
    if minSum < np.inf:
        # 0/0 case
        k = int(np.argmin(piTmp + omTmp))
        return x.copy(), k + 1

    ratios = pi / omega
    ratiosPos = ratios.copy()
    mask_pos = ~((pi >= -eps) & (omega > 0))
    ratiosPos[mask_pos] = np.inf
    jj2 = int(np.argmin(ratiosPos))

    ratiosNeg = ratios.copy()
    mask_neg = ~((pi <= eps) & (omega < 0))
    ratiosNeg[mask_neg] = -np.inf
    maxRNeg = np.max(ratiosNeg)
    jj3 = int(np.argmax(ratiosNeg))

    # Compute facet alphas analogous to LARGEfacetIntersection
    def facet_alpha(j_index: int) -> float:
        # j_index: 0-based
        if j_index == m - 1:
            # Remove last col and translate by p
            X = S[:, :m - 1] - p[:, None] @ np.ones((1, m - 1), dtype=S.dtype)
        else:
            # Remove j-th col from S_tilde translated by pm
            X = np.concatenate([S_bar[:, :j_index], S_bar[:, j_index + 1 :]], axis=1)
        if X.shape[1] == 0:
            return np.inf
        Qfac, _ = np.linalg.qr(X, mode='complete')
        Q_perp = Qfac[:, m - 1 :]
        temp = d @ Q_perp
        idx = np.where(np.abs(temp) > eps)[0]
        if idx.size == 0:
            return np.inf
        i0 = idx[0]
        alpha = ((p - x) @ Q_perp[:, i0]) / temp[i0]
        return float(alpha)

    alphas = [alpha_bis, np.inf, np.inf]
    flags = [0, np.inf, np.inf]
    alpha2 = facet_alpha(jj2)
    alphas[1] = alpha2
    flags[1] = jj2 + 1
    alpha3 = np.inf
    if maxRNeg > 0:
        alpha3 = facet_alpha(jj3)
    alphas[2] = alpha3
    flags[2] = jj3 + 1

    # Select minimum non-negative alpha
    alphas_arr = np.array(alphas, dtype=float)
    alphas_arr[alphas_arr < -eps] = np.inf
    jj = int(np.argmin(alphas_arr))
    alpha_best = float(alphas_arr[jj])
    flag = int(flags[jj])

    if trace_fn is not None:
        alpha_facet_min = None
        if np.isfinite(alphas_arr[1:]).any():
            alpha_facet_min = float(np.nanmin(alphas_arr[1:]))
        trace_fn({
            "event": "line_alphas",
            "alpha_bis": None if not np.isfinite(alpha_bis) else float(alpha_bis),
            "alpha_facet_min": alpha_facet_min,
            "alpha_best": None if not np.isfinite(alpha_best) else float(alpha_best),
            "flag": int(flag),
            "num_candidates": int(np.isfinite(alphas_arr[1:]).sum()),
        })

    if not np.isfinite(alpha_best):
        return x.copy(), 0
    x_new = x + alpha_best * d
    return x_new, flag


def ls_testing(A: np.ndarray, B: np.ndarray,
               epsTol: float = 1e-4, epsTol2: float = 1e-4, epsTol3: float = 1e-6,
               debug: bool = False, trace_fn=None
               ) -> Tuple[int, float, float]:
    """
    Determine whether point sets A and B are linearly separable (LS), and quantify LS degree.

    A: shape (n, m1), columns are points
    B: shape (n, m2), columns are points
    Returns: (LS, LS_Degree, elapsed_time)
    """
    start_time = time.time()

    # Use float32 like MATLAB single to reduce selection divergence
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    n = A.shape[0]
    sizeA = A.shape[1]
    sizeB = B.shape[1]

    # Initial two points p1 and p2 from pairs (A1,B1) and (A1,B2)
    diff = A[:, 0] - B[:, 0]
    norm_diff = np.linalg.norm(diff)
    if norm_diff == 0:
        norm_diff = 1.0
    p1 = diff / norm_diff

    diff = A[:, 0] - B[:, min(1, sizeB - 1)]
    norm_diff = np.linalg.norm(diff)
    if norm_diff == 0:
        norm_diff = 1.0
    p2 = diff / norm_diff

    # Support set S
    S = np.column_stack([p1, p2])
    x = 0.5 * (p1 + p2)
    z = 0.5 * np.sqrt((p1 - p2).T @ (p1 - p2))
    if debug and trace_fn is not None:
        _xnorm2 = float(x @ x)
        trace_fn({
            "event": "init",
            "n": int(n),
            "sizeA": int(sizeA),
            "sizeB": int(sizeB),
            "z": float(z),
            "||x||2": _xnorm2,
            "xnorm2": _xnorm2,
        })

    radii = []

    iter_idx = 0
    while True:
        # Find farthest p over all normalized differences without materializing all pairs
        maxdist = -np.inf
        best_iA = 0
        best_iB = 0

        # Compute distances in blocks over B to control memory if needed
        # Here we do full loops; users can adapt block size if necessary
        x_dot = x  # reused
        for iA in range(sizeA):
            a_col = A[:, iA][:, None]
            # Compute differences with all B at once
            diffs = a_col - B  # shape (n, sizeB)
            norms = np.linalg.norm(diffs, axis=0)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            p_mat = diffs / norms
            # distances = -2 * x^T p + ||p||^2, and ||p||=1
            dists = -2.0 * (x_dot @ p_mat) + 1.0
            # Flatten along B
            idx = int(np.argmax(dists))
            val = float(dists[idx])
            if val > maxdist:
                maxdist = val
                best_iA = iA
                best_iB = idx

        # Stopping condition
        if np.sqrt(maxdist + x @ x) < z + epsTol:
            break
        if debug and trace_fn is not None:
            trace_fn({
                "event": "select_p",
                "iter": int(iter_idx + 1),
                "best_iA": int(best_iA + 1),
                "best_iB": int(best_iB + 1),
                "maxdist": float(maxdist),
                "lhs_stop": float(np.sqrt(maxdist + x @ x)),
                "rhs_stop": float(z + epsTol),
            })

        # Build new p from best indices
        diff = A[:, best_iA] - B[:, best_iB]
        norm_diff = np.linalg.norm(diff)
        if norm_diff == 0:
            norm_diff = 1.0
        p = diff / norm_diff

        # Ensure {S, p} is affinely independent by possibly removing a point
        S = _large_update_s(S, p, x, epsTol, trace_fn=trace_fn if debug else None)

        # Line search loop (QR-based)
        while True:
            x_new, flag = _qr_line_search(x, S, p, epsTol, trace_fn=trace_fn if debug else None)
            if flag == 0:
                x = x_new
                z = np.linalg.norm(x - p)
                break
            else:
                # Drop facet index (convert from 1-based to 0-based)
                drop_idx = flag - 1
                mask = np.ones(S.shape[1], dtype=bool)
                mask[drop_idx] = False
                S = S[:, mask]
                x = x_new

        # Add p to support set (prepend to mimic MATLAB insertion)
        S = np.column_stack([p, S])

        radii.append(z)
        if len(radii) >= 6:
            recent_increases = np.diff(np.array(radii[-5:]))
            if np.all(np.abs(recent_increases) < epsTol2):
                break
        iter_idx += 1

    # Decide LS
    if z > 1.0 - epsTol3:
        LS = 0
    else:
        LS = 1

    LS_Degree = 1.0 - float(z)
    elapsed = time.time() - start_time
    return LS, LS_Degree, elapsed


__all__ = ["ls_testing"]


