"""
Grader Variance Test — verifies that CompositeGrader produces a spread of scores.

WHY THIS EXISTS:
    A grader that always returns the same value (or a very narrow range) is broken —
    it can't tell a good agent from a bad one. This script checks that the grader
    produces meaningful variance across many different episodes and patients.

WHAT IT DOES:
    Runs `n` episodes of the specified task, grading each with an EMPTY trajectory
    (i.e. no actions taken, episode ended immediately after reset). This isolates
    natural variance coming from patient diversity rather than agent behaviour.
    If scores vary across patients even with an empty trajectory, the grader is
    sensitive to patient-level differences, which is the expected behaviour.

HOW TO RUN:
    python scripts/check_variance.py

PASS CRITERIA:
    Standard deviation should be > 0.05. The validate_local.sh script enforces
    this with an explicit exit(1) if variance is too low.
"""

import sys
import os

# Allow importing asha_env and graders from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import statistics
from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader


def run_variance_test(task_id: str = "medium_consultation", n: int = 20):
    """
    Run n episodes with empty trajectories and report score statistics.

    Each episode is reset with a freshly generated patient, then immediately
    graded without taking any actions. This measures how much the grader score
    varies purely due to patient diversity (disease mix, demographics, season).

    Args:
        task_id: Which task difficulty to use. Defaults to "medium_consultation".
        n:       Number of episodes to run. More episodes give a stabler stdev estimate.
    """
    env = AshaEnv()
    grader = CompositeGrader()
    scores = []

    for i in range(n):
        # Start a new patient episode. No actions are taken — we grade immediately.
        env.reset(task_id)

        # env.get_state() exposes the ground-truth (true diagnosis, full patient dict)
        # needed by the grader. The trajectory will be empty at this point.
        state = env.get_state()
        patient = state["patient"]
        true_diagnosis = state["true_diagnosis"]
        trajectory = state["trajectory"]  # Empty list — no actions taken yet.

        # Grade the empty trajectory. The score reflects the grader's baseline output
        # for this patient (e.g. partial credit for not diagnosing an emergency).
        score = grader.grade(
            trajectory=trajectory,
            true_diagnosis=true_diagnosis,
            patient=patient,
        )
        scores.append(score)
        print(f"  Episode {i+1:02d}: score = {score:.4f}  (diagnosis={true_diagnosis})")

    # --- Summary statistics ---
    print("\n--- Grader Variance Summary ---")
    print(f"  N        : {n}")
    print(f"  Mean     : {statistics.mean(scores):.4f}")
    # stdev requires at least 2 data points.
    print(f"  Std Dev  : {statistics.stdev(scores) if n > 1 else 0:.4f}")
    print(f"  Min      : {min(scores):.4f}")
    print(f"  Max      : {max(scores):.4f}")
    print(f"  Variance : {statistics.variance(scores) if n > 1 else 0:.6f}")


if __name__ == "__main__":
    print(f"Running grader variance test (20 episodes of 'medium_consultation')...\n")
    run_variance_test(task_id="medium_consultation", n=20)