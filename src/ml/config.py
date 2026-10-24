"""Shared settings for the ML module (single place to change thresholds)."""
from pathlib import Path

# Project root = two folders above this file (src/ml/config.py -> project root).
# Using absolute paths means the code works no matter where you run it from.
BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES = [
    "attendance", "midterm_score", "assignments_avg", "quizzes_avg",
    "participation_score", "projects_score", "study_hours_per_week",
    "stress_level", "sleep_hours_per_night",
]

# Human-readable names shown in the UI
LABELS = {
    "attendance": "Attendance",
    "midterm_score": "Midterm marks",
    "assignments_avg": "Assignment marks",
    "quizzes_avg": "Quiz marks",
    "participation_score": "Class participation",
    "projects_score": "Project marks",
    "study_hours_per_week": "Study hours",
    "stress_level": "Stress level",
    "sleep_hours_per_night": "Sleep hours",
}

# Valid ranges, used by predict() to validate input
RANGES = {
    "attendance": (0, 100), "midterm_score": (0, 100), "assignments_avg": (0, 100),
    "quizzes_avg": (0, 100), "participation_score": (0, 100), "projects_score": (0, 100),
    "study_hours_per_week": (0, 168), "stress_level": (1, 10), "sleep_hours_per_night": (0, 24),
}

# Risk thresholds on the predicted total score
LOW_CUTOFF = 70        # score >= 70 -> Low risk
MODERATE_CUTOFF = 60   # 60 <= score < 70 -> Moderate, below 60 -> High
# Safety margin: shifts both cutoffs up so more struggling students are caught
# (missing an at-risk student is worse than a false alarm).
SAFETY_MARGIN = 3

TARGET = "total_score"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
