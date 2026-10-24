import pytest
from src.ml.predict import predict

GOOD = dict(attendance=92, midterm_score=85, assignments_avg=88, quizzes_avg=84,
            participation_score=90, projects_score=92, study_hours_per_week=20,
            stress_level=3, sleep_hours_per_night=7.5)
WEAK = dict(attendance=55, midterm_score=42, assignments_avg=45, quizzes_avg=40,
            participation_score=40, projects_score=45, study_hours_per_week=3,
            stress_level=9, sleep_hours_per_night=5)


def test_output_format():
    r = predict(GOOD)
    assert set(r) == {"score", "risk", "factors"}
    assert 0 <= r["score"] <= 100
    assert r["risk"] in {"Low", "Moderate", "High"}


def test_strong_student_is_low_risk():
    assert predict(GOOD)["risk"] == "Low"


def test_weak_student_is_high_risk():
    r = predict(WEAK)
    assert r["risk"] == "High"
    assert len(r["factors"]) >= 1


def test_weak_scores_lower_than_good():
    assert predict(WEAK)["score"] < predict(GOOD)["score"]


def test_missing_field_raises():
    bad = dict(GOOD); bad.pop("attendance")
    with pytest.raises(ValueError):
        predict(bad)


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        predict({**GOOD, "attendance": 150})


def test_non_numeric_raises():
    with pytest.raises(ValueError):
        predict({**GOOD, "midterm_score": "abc"})
