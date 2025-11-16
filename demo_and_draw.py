import os
import sys
import numpy as np

# Ensure imports work when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_generation import generate_nls_data, generate_ls_data
from ls_testing import ls_testing


def plot_sets(A: np.ndarray, B: np.ndarray, title: str = "") -> None:
    import matplotlib.pyplot as plt
    n = A.shape[0]
    if n == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(A[0, :], A[1, :], s=16, c="tab:blue", label="A", alpha=0.8, edgecolors="none")
        plt.scatter(B[0, :], B[1, :], s=16, c="tab:red", label="B", alpha=0.8, edgecolors="none")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    elif n == 3:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d proj)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(A[0, :], A[1, :], A[2, :], s=12, c="tab:blue", label="A", alpha=0.8)
        ax.scatter(B[0, :], B[1, :], B[2, :], s=12, c="tab:red", label="B", alpha=0.8)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        if title:
            ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Plotting is only supported for n=2 or n=3 (got n={n}). Skipping visualization.")


def main():
    # mode: "nls", "ls", or path to CSV file (first row labels)
    mode = "nls"
    n = 3  # set to 2 or 3 to enable plotting
    m = 100

    print(f"real Linear separability = {mode}")

    if mode == "nls":
        data = generate_nls_data(n, m)
    elif mode == "ls":
        data = generate_ls_data(n, m)
    elif isinstance(mode, str) and os.path.exists(mode):
        # CSV file
        data = np.loadtxt(mode, delimiter=",")
        n = data.shape[0] - 1
    else:
        raise ValueError("Invalid mode; use 'nls', 'ls', or a CSV file path.")

    labels = data[0, :]
    values = data[1:, :]
    A = values[:, labels == 0]
    B = values[:, labels == 1]

    LS, LS_Degree, elapsed = ls_testing(A.astype(float), B.astype(float))
    ls_text = "LS" if LS == 1 else "NLS"
    print(f"Linear separability = {ls_text}")
    print(f"LS_Degree = {LS_Degree:.6f}")
    print(f"Use time = {elapsed:.3f}")

    title = f"A (blue) vs B (red) — {ls_text}, LS_Degree={LS_Degree:.4f}"
    plot_sets(A, B, title=title)


if __name__ == "__main__":
    main()


