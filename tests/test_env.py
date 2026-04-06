"""
Tests for AshaEnv — the core RL environment (asha_env/env.py).

These tests verify the environment's public API contract:
    - reset() returns a correctly structured observation
    - step() returns the right types and terminates correctly
    - Terminal actions (diagnose, refer) end the episode
    - Max steps limit is respected
    - Invalid inputs raise the right errors
    - get_state() exposes ground-truth fields for grading

Run with:
    pytest tests/test_env.py -v
"""

import sys
import os

# Allow importing asha_env from the project root when running tests directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from asha_env.env import AshaEnv


# Shared env fixture — creates a fresh AshaEnv instance for every test function.
# Using a fixture (rather than a module-level global) ensures tests are isolated.
@pytest.fixture
def env():
    return AshaEnv()


def test_reset_returns_valid_observation(env):
    """reset() must return a dict with the four top-level keys the agent needs."""
    obs = env.reset("medium_consultation")
    assert "patient" in obs
    assert "asha_context" in obs
    assert "available_actions" in obs
    assert "episode_id" in obs
    # episode_id is generated as "ep_<12 hex chars>" — verify the prefix.
    assert obs["episode_id"].startswith("ep_")


def test_reset_observation_has_required_patient_fields(env):
    """The patient sub-dict must contain all fields the agent uses for reasoning."""
    obs = env.reset("medium_consultation")
    p = obs["patient"]
    assert "age" in p
    assert "gender" in p
    assert "chief_complaint" in p
    assert "location" in p
    assert "socioeconomic" in p
    assert "known_symptoms" in p      # Symptoms revealed so far (starts empty on medium/hard)
    assert "vitals" in p
    assert "revealed_history" in p    # History items revealed so far
    assert "village_context" in p     # Season, outbreaks, water source


def test_reset_observation_has_required_asha_fields(env):
    """The asha_context sub-dict must contain all fields needed for episode management."""
    obs = env.reset("medium_consultation")
    ctx = obs["asha_context"]
    assert "season" in ctx
    assert "kit_available" in ctx     # List of medicines in the ASHA kit
    assert "tests_available" in ctx   # List of tests available
    assert "patient_trust" in ctx
    assert "steps_taken" in ctx
    assert "max_steps" in ctx
    # At episode start: no steps taken and trust at its initial value.
    assert ctx["steps_taken"] == 0
    assert ctx["patient_trust"] == 75


def test_step_returns_reward_and_done(env):
    """step() must return the standard RL (obs, reward, done, info) tuple with correct types."""
    obs = env.reset("easy_diagnosis")
    action = obs["available_actions"][0]
    obs2, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert isinstance(obs2, dict)


def test_episode_ends_after_max_steps(env):
    """
    Deliberately avoid terminal actions (diagnose/refer) and only use ask_symptom
    to force the episode to run until the max_steps limit is hit.
    The safety cap of 100 prevents an infinite loop if the logic is broken.
    """
    obs = env.reset("easy_diagnosis")
    done = False
    steps = 0
    while not done:
        available = obs.get("available_actions", [])
        if not available:
            break
        # Prefer ask_symptom to avoid accidentally triggering a terminal action.
        action = None
        for a in available:
            if a.startswith("ask_symptom:"):
                action = a
                break
        if action is None:
            action = available[0]
        obs, reward, done, info = env.step(action)
        steps += 1
        if steps > 100:  # Safety cap — test should never reach this.
            break
    assert done
    assert steps <= env.task.max_steps


def test_available_actions_never_empty_before_done(env):
    """
    The agent should never face an empty action list mid-episode.
    Take 5 steps and verify available_actions is non-empty before each one.
    """
    obs = env.reset("medium_consultation")
    for _ in range(5):
        available = obs.get("available_actions", [])
        assert len(available) > 0
        action = available[0]
        obs, reward, done, info = env.step(action)
        if done:
            break


def test_all_three_tasks(env):
    """All three task difficulty strings must be accepted by reset() without error."""
    for task_id in ["easy_diagnosis", "medium_consultation", "hard_complex_case"]:
        obs = env.reset(task_id)
        assert obs["episode_id"].startswith("ep_")
        assert len(obs["available_actions"]) > 0


def test_invalid_task_raises(env):
    """reset() must raise ValueError for an unrecognised task_id."""
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


def test_diagnose_ends_episode(env):
    """A diagnose: action is terminal — done must be True immediately after."""
    obs = env.reset("easy_diagnosis")
    diag_actions = [a for a in obs["available_actions"] if a.startswith("diagnose:")]
    assert len(diag_actions) > 0
    obs, reward, done, info = env.step(diag_actions[0])
    assert done


def test_refer_ends_episode(env):
    """A refer: action is terminal — done must be True immediately after."""
    obs = env.reset("easy_diagnosis")
    refer_actions = [a for a in obs["available_actions"] if a.startswith("refer:")]
    assert len(refer_actions) > 0
    obs, reward, done, info = env.step(refer_actions[0])
    assert done


def test_step_after_done_returns_zero_reward(env):
    """
    Calling step() after the episode has ended must return reward=0.0 and done=True.
    Agents that keep stepping after done must not receive accidental rewards.
    """
    obs = env.reset("easy_diagnosis")
    diag = [a for a in obs["available_actions"] if a.startswith("diagnose:")][0]
    env.step(diag)  # End the episode with a diagnose action.
    # Now try to take another step — should be silently ignored with zero reward.
    obs2, reward, done, info = env.step("ask_symptom:fever")
    assert done
    assert reward == 0.0


def test_get_state_has_true_diagnosis(env):
    """
    get_state() must expose the hidden ground-truth fields needed by graders.
    true_diagnosis and true_symptoms are not visible in the normal observation.
    """
    env.reset("medium_consultation")
    state = env.get_state()
    assert "true_diagnosis" in state
    assert state["true_diagnosis"] is not None   # A disease must have been assigned.
    assert "true_symptoms" in state
    assert len(state["true_symptoms"]) > 0       # Every disease has at least one symptom.


def test_episode_id_mismatch_not_accepted(env):
    """
    The env itself stores the active episode_id. This test verifies it matches
    what was returned by reset() — the server uses this to reject stale requests.
    Note: episode_id validation against the request is enforced in server/app.py,
    not inside AshaEnv directly.
    """
    obs = env.reset("easy_diagnosis")
    real_id = obs["episode_id"]
    assert env.episode_id == real_id
