"""Random baseline agent — picks from available_actions uniformly at random."""

import random
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader

TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]
EPISODES_PER_TASK = 10


def run_random_episode(env: AshaEnv, grader: CompositeGrader, task_id: str) -> float:
    obs = env.reset(task_id)
    done = False
    trajectory = []

    while not done:
        available = obs.get("available_actions", [])
        if not available:
            break
        action = random.choice(available)
        obs, reward, done, info = env.step(action)
        trajectory.append(action)

    state = env.get_state()
    return grader.grade(trajectory, state["true_diagnosis"], state["patient"])


def main():
    env = AshaEnv()
    grader = CompositeGrader()
    results = {}

    for task_id in TASKS:
        scores = []
        for _ in range(EPISODES_PER_TASK):
            score = run_random_episode(env, grader, task_id)
            scores.append(score)
        results[task_id] = scores

    # Print results
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
        sd = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"{task_id:<25} | {len(scores):>8} | {avg:>6.3f} | {mn:>6.3f} | {mx:>6.3f} | {sd:>6.3f}")

    print(f"{'='*60}")
    print(f"Overall: mean={statistics.mean(all_scores):.3f}, stdev={statistics.stdev(all_scores):.3f}")


if __name__ == "__main__":
    main()
