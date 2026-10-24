"""Converts a predicted score into a risk level."""
from src.ml.config import LOW_CUTOFF, MODERATE_CUTOFF, SAFETY_MARGIN


def score_to_risk(score, margin=SAFETY_MARGIN):
    if score >= LOW_CUTOFF + margin:
        return "Low"
    if score >= MODERATE_CUTOFF + margin:
        return "Moderate"
    return "High"
