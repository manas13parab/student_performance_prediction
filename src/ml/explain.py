"""
explain.py - which inputs push a student's predicted score up or down?

Works with any model: for each feature we replace the student's value with the
class average and see how much the prediction changes.
    impact = prediction(student) - prediction(student with that feature = average)
Negative impact = this feature is pulling the score DOWN (a risk factor).
"""
import pandas as pd
from src.ml.config import FEATURES, LABELS


def feature_impacts(bundle, student: dict) -> dict:
    pipe, means = bundle["pipeline"], bundle["feature_means"]
    row = pd.DataFrame([{f: student[f] for f in FEATURES}])
    base = float(pipe.predict(row)[0])
    impacts = {}
    for f in FEATURES:
        tweaked = row.copy()
        tweaked[f] = means[f]
        impacts[f] = base - float(pipe.predict(tweaked)[0])
    return impacts


def top_risk_factors(bundle, student: dict, n=3):
    """Return up to n [(readable_name, impact_in_marks)] hurting the score most."""
    impacts = feature_impacts(bundle, student)
    negatives = sorted(((f, v) for f, v in impacts.items() if v < -0.5), key=lambda x: x[1])
    return [(LABELS[f], round(v, 2)) for f, v in negatives[:n]]
