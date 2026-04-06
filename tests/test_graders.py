import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import random
from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader
from graders.diagnosis_grader import DiagnosisGrader
from graders.safety_grader import SafetyGrader
from graders.efficiency_grader import EfficiencyGrader
from graders.referral_grader import ReferralGrader


@pytest.fixture
def env():
    return AshaEnv()


@pytest.fixture
def grader():
    return CompositeGrader()


def _run_random_episode(env, task_id="medium_consultation"):
    obs = env.reset(task_id)
    trajectory = []
    done = False
    while not done:
        action = random.choice(obs.get("available_actions", ["diagnose:malaria"]))
        obs, reward, done, info = env.step(action)
        trajectory.append(action)
    state = env.get_state()
    return trajectory, state["true_diagnosis"], state["patient"]


def test_diagnosis_grader_returns_float_between_0_and_1(env):
    g = DiagnosisGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_safety_grader_returns_float_between_0_and_1(env):
    g = SafetyGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_efficiency_grader_returns_float_between_0_and_1(env):
    g = EfficiencyGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_referral_grader_returns_float_between_0_and_1(env):
    g = ReferralGrader()
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = g.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_composite_grader_returns_float_between_0_and_1(env, grader):
    for _ in range(10):
        traj, diag, patient = _run_random_episode(env)
        score = grader.grade(traj, diag, patient)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_composite_grader_not_constant(env, grader):
    """Run 20 episodes and verify scores have variance > 0."""
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
    """Pre-eclampsia is an emergency — missing referral should score low."""
    g = SafetyGrader()
    # Simulate trajectory that just asks questions and never refers
    trajectory = ["ask_symptom:fever", "ask_symptom:headache", "diagnose:hypertension"]
    patient = {"disease_data": {"emergency": True, "severity": "critical", "referral_level": "district_hospital"}}
    score = g.grade(trajectory, "pre_eclampsia", patient)
    assert score <= 0.3


def test_efficiency_grader_rewards_fewer_questions(env):
    g = EfficiencyGrader()
    # Short trajectory with correct diagnosis
    short_traj = ["ask_symptom:fever", "ask_symptom:chills", "diagnose:malaria"]
    # Long trajectory with correct diagnosis
    long_traj = [
        "ask_symptom:fever", "ask_symptom:chills", "ask_symptom:headache",
        "ask_symptom:nausea", "ask_symptom:rash", "ask_symptom:diarrhoea",
        "ask_symptom:cough", "ask_symptom:fatigue", "ask_symptom:weakness",
        "diagnose:malaria",
    ]
    patient = {}
    short_score = g.grade(short_traj, "malaria", patient)
    long_score = g.grade(long_traj, "malaria", patient)
    assert short_score >= long_score, f"Short ({short_score}) should score >= Long ({long_score})"


def test_diagnosis_grader_correct_vs_wrong(env):
    g = DiagnosisGrader()
    patient = {}
    correct = g.grade(["diagnose:malaria"], "malaria", patient)
    wrong = g.grade(["diagnose:diabetes"], "malaria", patient)
    assert correct > wrong
