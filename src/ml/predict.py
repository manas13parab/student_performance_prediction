"""
predict.py - the function the app calls.

    from src.ml.predict import predict
    predict({...9 fields...})
    -> {"score": 61.4, "risk": "High",
        "factors": [("Attendance", -3.2), ("Project marks", -2.1)]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # works from any folder

import joblib
import pandas as pd
from src.ml.config import FEATURES, MODEL_PATH, RANGES
from src.ml.explain import top_risk_factors
from src.ml.risk import score_to_risk

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def validate(student: dict):
    missing = [f for f in FEATURES if f not in student]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    for f in FEATURES:
        lo, hi = RANGES[f]
        try:
            v = float(student[f])
        except (TypeError, ValueError):
            raise ValueError(f"{f} must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"{f} must be between {lo} and {hi}")


def predict(student: dict) -> dict:
    validate(student)
    student = {f: float(student[f]) for f in FEATURES}
    bundle = _load()
    row = pd.DataFrame([student])
    score = float(bundle["pipeline"].predict(row)[0])
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "risk": score_to_risk(score),
        "factors": top_risk_factors(bundle, student),
    }


if __name__ == "__main__":
    # Quick manual check:  python src/ml/predict.py
    demo = dict(attendance=58, midterm_score=45, assignments_avg=50, quizzes_avg=48,
                participation_score=40, projects_score=52, study_hours_per_week=4,
                stress_level=8, sleep_hours_per_night=5)
    print(predict(demo))
