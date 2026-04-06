"""
Tests for all graders (graders/).

These tests verify two things:
    1. CONTRACT  — every grader always returns a float in [0.0, 1.0].
    2. BEHAVIOUR — graders rank trajectories in the expected order
                   (e.g. correct diagnosis > wrong diagnosis, short > long).

A grader that always returns the same value would pass the contract tests but fail
the behaviour tests. Both sets are needed to catch different classes of bugs.

Run with:
    pytest tests/test_graders.py -v
"""

import sys
import os

# Allow importing asha_env and graders from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import random
from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader
from graders.diagnosis_grader import DiagnosisGrader
from graders.safety_grader import SafetyGrader
from graders.efficiency_grader import EfficiencyGrader
from graders.referral_grader import ReferralGrader


# Shared fixtures — fresh instances for every test.
@pytest.fixture
def env():
    return AshaEnv()

@pytest.fixture
def grader():
    return CompositeGrader()


def _run_random_episode(env, task_id="medium_consultation"):
    """
    Helper: run one full random episode and return the data needed by graders.

    Picks actions uniformly at random until done=True, then fetches ground-truth
    from get_state(). Used by the contract tests to get realistic trajectories
    without needing to hard-code them.

    Returns:
        (trajectory, true_diagnosis, patient) — the three args every grader needs.
    """
    obs = env.reset(task_id)
    trajectory = []
    done = False
    while not done:
        # Fallback action ensures the loop always terminates even if available is empty.
        action = random.choice(obs.get("available_actions", ["diagnose:severe_anaemia"]))
        obs, reward, done, info = env.step(action)
        trajectory.append(action)
    state = env.get_state()
    return trajectory, state["true_diagnosis"], state["patient"]


# ---------------------------------------------------------------------------
# Contract tests — each grader must return float in [0.0, 1.0] on any trajectory
# ---------------------------------------------------------------------------

def test_diagnosis_grader_returns_float_between_0_and_1(env):
    """DiagnosisGrader output must be a float in [0.0, 1.0] across 10 random episodes."""
    g = DiagnosisGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_safety_grader_returns_float_between_0_and_1(env):
    """SafetyGrader output must be a float in [0.0, 1.0] across 10 random episodes."""
    g = SafetyGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_efficiency_grader_returns_float_between_0_and_1(env):
    """EfficiencyGrader output must be a float in [0.0, 1.0] across 10 random episodes."""
    g = EfficiencyGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_referral_grader_returns_float_between_0_and_1(env):
    """ReferralGrader output must be a float in [0.0, 1.0] across 10 random episodes."""
    g = ReferralGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_composite_grader_returns_float_between_0_and_1(env, grader):
    """CompositeGrader output must be a float in [0.0, 1.0] across 10 random episodes."""
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = grader.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Behaviour tests — graders must rank trajectories in clinically sensible order
# ---------------------------------------------------------------------------

def test_composite_grader_not_constant(env, grader):
    """
    Run 20 random episodes across all task types and confirm the composite grader
    produces genuinely varying scores (not a constant or near-constant output).

    Two checks:
        1. More than one unique score value (absolute minimum).
        2. Standard deviation > 0.01 (meaningful spread).

    A grader that always returns the same value is broken — this test catches that.
    """
    scores = []
    for _ in range(20):
        task = random.choice(["easy_diagnosis", "medium_consultation", "hard_complex_case"])
        traj, diag, patient = _run_random_episode(env, task)
        score = grader.grade(traj, diag, patient)
        scores.append(score)

    unique_scores = set(round(s, 4) for s in scores)
    assert len(unique_scores) > 1, f"Grader returned constant scores: {scores[:5]}"

    import statistics
    stdev = statistics.stdev(scores)
    assert stdev > 0.01, f"Score variance too low: stdev={stdev:.4f}"


def test_safety_grader_penalizes_missed_emergency(env):
    """
    Eclampsia is flagged as an emergency in diseases.json. A trajectory that
    diagnoses wrong (severe_anaemia) without referring should score very low
    because the patient needed urgent referral to district_hospital.
    Expected: score <= 0.3.
    """
    g = SafetyGrader()
    # Trajectory: asked some symptoms, then made the wrong diagnosis with no referral.
    trajectory = ["ask_symptom:fever", "ask_symptom:severe_headache", "diagnose:severe_anaemia"]
    # Minimal patient dict with just the fields SafetyGrader reads from it.
    patient = {"disease_data": {"emergency": True, "severity": "critical", "referral_level": "district_hospital"}}
    score = g.grade(trajectory, "eclampsia", patient)
    assert score <= 0.3


def test_efficiency_grader_rewards_fewer_questions(env):
    """
    Two trajectories that both correctly diagnose severe_anaemia — one with 2
    questions, one with 9. The shorter trajectory should score >= the longer one.
    This confirms the grader penalises unnecessary questioning.
    """
    g = EfficiencyGrader()
    # 2 questions + correct diagnosis.
    short_traj = ["ask_symptom:pallor", "ask_symptom:fatigue", "diagnose:severe_anaemia"]
    # 9 questions (many irrelevant) + correct diagnosis.
    long_traj = [
        "ask_symptom:pallor", "ask_symptom:fatigue", "ask_symptom:weakness",
        "ask_symptom:dizziness", "ask_symptom:high_bp", "ask_symptom:fever",
        "ask_symptom:nausea", "ask_symptom:vaginal_bleeding", "ask_symptom:convulsions",
        "diagnose:severe_anaemia",
    ]
    patient = {}  # EfficiencyGrader loads disease data itself from diseases.json.
    short_score = g.grade(short_traj, "severe_anaemia", patient)
    long_score = g.grade(long_traj, "severe_anaemia", patient)
    assert short_score >= long_score, f"Short ({short_score}) should score >= Long ({long_score})"


def test_diagnosis_grader_correct_vs_wrong(env):
    """
    A trajectory that diagnoses correctly must score higher than one that
    diagnoses the wrong disease for the same true diagnosis.
    """
    g = DiagnosisGrader()
    patient = {}  # DiagnosisGrader loads disease data itself from diseases.json.
    correct = g.grade(["diagnose:severe_anaemia"], "severe_anaemia", patient)
    wrong = g.grade(["diagnose:gestational_diabetes"], "severe_anaemia", patient)
    assert correct > wrong
