import os
import sys
import numpy as np

# Ensure imports work when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_generation import generate_nls_data, generate_ls_data
from ls_testing import ls_testing


def main():
    # mode: "nls", "ls", or path to CSV file (first row labels)
    mode = "nls"
    n = 50
    m = 1000

    if mode == "nls":
        data = generate_nls_data(n, m)
    elif mode == "ls":
        data = generate_ls_data(n, m)
    elif isinstance(mode, str) and os.path.exists(mode):
        # CSV file
        data = np.loadtxt(mode, delimiter=",")
    else:
        raise ValueError("Invalid mode; use 'nls', 'ls', or a CSV file path.")

    labels = data[0, :]
    values = data[1:, :]
    A = values[:, labels == 0]
    B = values[:, labels == 1]

    LS, LS_Degree, elapsed = ls_testing(A.astype(float), B.astype(float))
    print(f"Linear separability = {'LS' if LS == 1 else 'NLS'}")
    print(f"LS_Degree = {LS_Degree:.6f}")
    print(f"Use time = {elapsed:.3f}")


if __name__ == "__main__":
    main()


