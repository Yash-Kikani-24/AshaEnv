import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from asha_env.env import AshaEnv


@pytest.fixture
def env():
    return AshaEnv()


def test_reset_returns_valid_observation(env):
    obs = env.reset("medium_consultation")
    assert "patient" in obs
    assert "asha_context" in obs
    assert "available_actions" in obs
    assert "episode_id" in obs
    assert obs["episode_id"].startswith("ep_")


def test_reset_observation_has_required_patient_fields(env):
    obs = env.reset("medium_consultation")
    p = obs["patient"]
    assert "age" in p
    assert "gender" in p
    assert "chief_complaint" in p
    assert "location" in p
    assert "socioeconomic" in p
    assert "known_symptoms" in p
    assert "vitals" in p
    assert "revealed_history" in p
    assert "village_context" in p


def test_reset_observation_has_required_asha_fields(env):
    obs = env.reset("medium_consultation")
    ctx = obs["asha_context"]
    assert "season" in ctx
    assert "kit_available" in ctx
    assert "tests_available" in ctx
    assert "patient_trust" in ctx
    assert "steps_taken" in ctx
    assert "max_steps" in ctx
    assert ctx["steps_taken"] == 0
    assert ctx["patient_trust"] == 75


def test_step_returns_reward_and_done(env):
    obs = env.reset("easy_diagnosis")
    action = obs["available_actions"][0]
    obs2, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "observation" not in obs2 or isinstance(obs2, dict)


def test_episode_ends_after_max_steps(env):
    obs = env.reset("easy_diagnosis")
    done = False
    steps = 0
    while not done:
        available = obs.get("available_actions", [])
        if not available:
            break
        # Pick a non-terminal action to force max steps
        action = None
        for a in available:
            if a.startswith("ask_symptom:"):
                action = a
                break
        if action is None:
            action = available[0]
        obs, reward, done, info = env.step(action)
        steps += 1
        if steps > 100:
            break
    assert done
    assert steps <= env.task.max_steps


def test_available_actions_never_empty_before_done(env):
    obs = env.reset("medium_consultation")
    for _ in range(5):
        available = obs.get("available_actions", [])
        assert len(available) > 0
        action = available[0]
        obs, reward, done, info = env.step(action)
        if done:
            break


def test_all_three_tasks(env):
    for task_id in ["easy_diagnosis", "medium_consultation", "hard_complex_case"]:
        obs = env.reset(task_id)
        assert obs["episode_id"].startswith("ep_")
        assert len(obs["available_actions"]) > 0


def test_invalid_task_raises(env):
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


def test_diagnose_ends_episode(env):
    obs = env.reset("easy_diagnosis")
    diag_actions = [a for a in obs["available_actions"] if a.startswith("diagnose:")]
    assert len(diag_actions) > 0
    obs, reward, done, info = env.step(diag_actions[0])
    assert done


def test_refer_ends_episode(env):
    obs = env.reset("easy_diagnosis")
    refer_actions = [a for a in obs["available_actions"] if a.startswith("refer:")]
    assert len(refer_actions) > 0
    obs, reward, done, info = env.step(refer_actions[0])
    assert done


def test_step_after_done_returns_zero_reward(env):
    obs = env.reset("easy_diagnosis")
    diag = [a for a in obs["available_actions"] if a.startswith("diagnose:")][0]
    env.step(diag)
    obs2, reward, done, info = env.step("ask_symptom:fever")
    assert done
    assert reward == 0.0


def test_get_state_has_true_diagnosis(env):
    env.reset("medium_consultation")
    state = env.get_state()
    assert "true_diagnosis" in state
    assert state["true_diagnosis"] is not None
    assert "true_symptoms" in state
    assert len(state["true_symptoms"]) > 0


def test_episode_id_mismatch_not_accepted(env):
    obs = env.reset("easy_diagnosis")
    real_id = obs["episode_id"]
    # Env step doesn't check episode_id itself, server does
    # But we can verify episode_id is set
    assert env.episode_id == real_id
