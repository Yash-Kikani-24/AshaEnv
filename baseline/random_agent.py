"""
Random baseline agent for the ASHA environment.

This agent picks an action uniformly at random from the available_actions list
on every step. It makes no use of any clinical knowledge or patient information.

Purpose:
    Establish a floor score — any meaningful agent should beat this baseline.
    If your agent scores near the random baseline, it is essentially not learning.

Usage:
    python baseline/random_agent.py

Output:
    A results table showing avg/min/max/stdev scores across all three task difficulties.
    Scores are produced by CompositeGrader (0.0–1.0), which weights diagnosis,
    safety, efficiency, and referral accuracy.
"""

import random
import statistics
import sys
import os

# Allow importing asha_env and graders from the project root when running this
# script directly (i.e., python baseline/random_agent.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader

# The three task difficulties defined in AshaEnv. Each controls which diseases
# can appear and how many steps the agent gets.
TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]

# How many independent episodes to run per task. More episodes give a more
# stable average but take longer. 10 is enough for a quick benchmark.
EPISODES_PER_TASK = 10


def run_random_episode(env: AshaEnv, grader: CompositeGrader, task_id: str) -> float:
    """
    Run one episode with the random agent and return the composite score.

    The agent loops until the environment signals done=True (either a terminal
    action like diagnose/refer was taken, or max_steps was reached). At each
    step it picks one action at random from obs["available_actions"].

    Args:
        env:     The AshaEnv instance (reused across episodes).
        grader:  CompositeGrader that scores the full trajectory after the episode.
        task_id: One of TASKS — controls difficulty and allowed diseases.

    Returns:
        A float in [0.0, 1.0] representing the composite grader score for the episode.
    """
    # Start a fresh patient episode and get the initial observation.
    obs = env.reset(task_id)
    done = False
    trajectory = []  # Ordered list of actions taken (needed by the grader).

    while not done:
        available = obs.get("available_actions", [])
        if not available:
            # No actions left — environment is stuck. Bail out.
            break

        # Random policy: pick any valid action with equal probability.
        action = random.choice(available)
        obs, reward, done, info = env.step(action)
        trajectory.append(action)

    # env.get_state() exposes hidden ground-truth (true diagnosis, full patient
    # data) used by the grader to score the trajectory.
    state = env.get_state()
    return grader.grade(trajectory, state["true_diagnosis"], state["patient"])


def main():
    """
    Benchmark the random agent across all task difficulties and print a summary table.

    For each task, EPISODES_PER_TASK independent episodes are run. The scores
    are collected and summarised (mean, min, max, stdev) and printed to stdout.
    An overall aggregate across all tasks is printed at the end.
    """
    env = AshaEnv()
    grader = CompositeGrader()

    # results[task_id] = list of float scores, one per episode.
    results = {}

    for task_id in TASKS:
        scores = []
        for _ in range(EPISODES_PER_TASK):
            score = run_random_episode(env, grader, task_id)
            scores.append(score)
        results[task_id] = scores

    # --- Print results table ---
    print(f"\n{'='*60}")
    print("Random Agent Baseline")
    print(f"{'='*60}")
    print(f"{'Task':<25} | {'Episodes':>8} | {'Avg':>6} | {'Min':>6} | {'Max':>6} | {'Stdev':>6}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    all_scores = []
    for task_id in TASKS:
        scores = results[task_id]
        all_scores.extend(scores)
        avg = statistics.mean(scores)
        mn = min(scores)
        mx = max(scores)
        # stdev requires at least 2 data points.
        sd = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"{task_id:<25} | {len(scores):>8} | {avg:>6.3f} | {mn:>6.3f} | {mx:>6.3f} | {sd:>6.3f}")

    print(f"{'='*60}")
    print(f"Overall: mean={statistics.mean(all_scores):.3f}, stdev={statistics.stdev(all_scores):.3f}")


if __name__ == "__main__":
    main()
