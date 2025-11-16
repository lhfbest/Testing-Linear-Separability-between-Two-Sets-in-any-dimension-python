import os
import sys
import json
from typing import Optional
import numpy as np

# Ensure imports work when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from ls_testing import ls_testing


def trace_to_file(path):
    f = open(path, "w", encoding="utf-8")
    def _fn(e):
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
        f.flush()
    return _fn


def main(csv_path: Optional[str] = None, out_trace: Optional[str] = None):
    if csv_path is None:
        # Default to the provided LS CSV example in the MATLAB folder
        csv_path = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, "Ours_LS_Testing_code_matlab", "Dimension10_Size5000_LS.csv"))
    if out_trace is None:
        out_trace = os.path.join(CURRENT_DIR, "py_trace.jsonl")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",").astype(np.float32)
    labels = data[0, :]
    values = data[1:, :]
    A = values[:, labels == 0].astype(np.float32)
    B = values[:, labels == 1].astype(np.float32)

    tracer = trace_to_file(out_trace)
    LS, LS_Degree, elapsed = ls_testing(A.astype(float), B.astype(float), debug=True, trace_fn=tracer)
    print(f"Linear separability = {'LS' if LS == 1 else 'NLS'}")
    print(f"LS_Degree = {LS_Degree:.6f}")
    print(f"Use time = {elapsed:.3f}")
    print(f"Trace written to: {out_trace}")


if __name__ == "__main__":
    # Optional CLI args: csv_path, out_trace
    argv = sys.argv[1:]
    csv = argv[0] if len(argv) >= 1 else None
    out = argv[1] if len(argv) >= 2 else None
    main(csv_path=csv, out_trace=out)


