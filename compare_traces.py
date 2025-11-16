import json
import math
import sys
from typing import Any, Dict, List, Tuple


EVENTS_TO_COMPARE = {
    "init": {
        "fields": ["n", "sizeA", "sizeB"],
        "float_fields": ["z", "||x||2"],
    },
    "select_p": {
        "fields": ["iter", "best_iA", "best_iB"],
        "float_fields": ["maxdist", "lhs_stop", "rhs_stop"],
    },
    "updateS_remove": {
        "fields": ["m_before", "remove_index"],
        "float_fields": ["min_ratio"],
    },
    "line_dir": {
        "fields": ["m"],
        "float_fields": ["||d||", "dot_pmmp_d"],
    },
    "line_alphas": {
        "fields": ["flag", "num_candidates"],
        "float_fields": ["alpha_bis", "alpha_facet_min", "alpha_best"],
    },
}


def approx_equal(a: Any, b: Any, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return a == b
    if math.isfinite(af) and math.isfinite(bf):
        return math.isclose(af, bf, abs_tol=atol, rel_tol=rtol)
    return (not math.isfinite(af)) and (not math.isfinite(bf))


def load_trace(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "event" in obj:
                    events.append(obj)
            except Exception:
                continue
    return events


def compare_traces(py_events: List[Dict[str, Any]],
                   ml_events: List[Dict[str, Any]],
                   max_mismatches: int = 50) -> int:
    # Filter events to a comparable subset and align by order
    def filt(ev: Dict[str, Any]) -> bool:
        return ev.get("event") in EVENTS_TO_COMPARE

    py = [e for e in py_events if filt(e)]
    ml = [e for e in ml_events if filt(e)]

    n = min(len(py), len(ml))
    mismatches = 0
    i = 0
    while i < n and mismatches < max_mismatches:
        e1 = py[i]
        e2 = ml[i]
        name1 = e1.get("event")
        name2 = e2.get("event")
        if name1 != name2:
            print(f"[#{i}] Event mismatch: py={name1}, ml={name2}")
            mismatches += 1
            i += 1
            continue
        spec = EVENTS_TO_COMPARE[name1]
        ok = True
        # Exact fields
        for k in spec.get("fields", []):
            if e1.get(k) != e2.get(k):
                print(f"[#{i}] Field '{k}' mismatch: py={e1.get(k)}, ml={e2.get(k)}")
                ok = False
        # Float fields with tolerance
        for k in spec.get("float_fields", []):
            if not approx_equal(e1.get(k), e2.get(k)):
                print(f"[#{i}] Float '{k}' mismatch: py={e1.get(k)}, ml={e2.get(k)}")
                ok = False
        if ok:
            print(f"[#{i}] {name1}: OK")
        else:
            mismatches += 1
        i += 1

    if len(py) != len(ml):
        print(f"Length differs after filtering: py={len(py)}, ml={len(ml)}")
        mismatches += 1

    return mismatches


def main(py_trace_path: str, ml_trace_path: str) -> None:
    py_events = load_trace(py_trace_path)
    ml_events = load_trace(ml_trace_path)
    mismatches = compare_traces(py_events, ml_events)
    if mismatches == 0:
        print("All compared events match within tolerances.")
    else:
        print(f"Found {mismatches} mismatches. Review logs above.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_traces.py <py_trace.jsonl> <matlab_trace.jsonl>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])


