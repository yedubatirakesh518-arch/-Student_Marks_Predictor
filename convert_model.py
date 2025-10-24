"""Utility: try to load a .pkl with several loaders and re-save as joblib.

Usage:
    python convert_model.py student_marks_predictor.pkl converted_model.joblib

This will try joblib, pickle (with latin1 fallback), and dill to load the input file.
If successful, it writes the object with joblib.dump to the output path.
"""
import sys
from pathlib import Path
import pickle
import joblib
import dill


def load_model_from_path(path):
    path = Path(path)
    last_exc = None
    try:
        return joblib.load(path)
    except Exception as e:
        last_exc = e

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_exc = e

    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception as e:
        last_exc = e

    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        last_exc = e

    raise last_exc


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_model.py <input.pkl> <output.joblib>")
        sys.exit(2)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not inp.exists():
        print(f"Input file not found: {inp}")
        sys.exit(2)

    try:
        print(f"Trying to load {inp}...")
        m = load_model_from_path(inp)
        print("Loaded successfully. Re-saving with joblib...")
        joblib.dump(m, out)
        print(f"Saved to {out}")
    except Exception as e:
        print(f"Failed to load and convert model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
