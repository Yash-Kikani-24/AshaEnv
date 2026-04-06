#!/bin/bash
# validate_local.sh — full local validation pipeline for the ASHA environment.
#
# Run this script before committing or deploying to catch problems early.
# It runs all 5 checks in sequence and stops immediately on the first failure
# (because of `set -e` — any non-zero exit code aborts the script).
#
# Usage:
#   bash scripts/validate_local.sh
#
# Steps:
#   1. pytest test suite       — unit/integration correctness
#   2. Random agent baseline   — smoke-test that the env runs end-to-end
#   3. Rule-based baseline     — smoke-test that rule logic still works
#   4. Grader variance check   — ensures graders are not returning constant scores
#   5. Docker build (optional) — verifies the container builds cleanly
#
# Exit codes:
#   0 — all checks passed
#   1 — at least one check failed (the failing step is shown in the output)

set -e  # Abort immediately if any command returns a non-zero exit code.

echo "============================================"
echo "ASHA Agent Environment — Local Validation"
echo "============================================"
echo ""

# --- Step 1: Unit and integration tests ---
# Runs everything in tests/ with verbose output so failures are easy to locate.
echo "[1/5] Running tests..."
python -m pytest tests/ -v
echo ""

# --- Step 2: Random agent baseline ---
# Runs 10 episodes per task with a purely random policy. Confirms the environment
# resets, steps, and grades without crashing. Scores should be low (near-random).
echo "[2/5] Running random baseline..."
python baseline/random_agent.py
echo ""

# --- Step 3: Rule-based agent baseline ---
# Runs 10 episodes per task with the handcrafted clinical rules. Confirms the
# rule logic still executes correctly after any recent changes.
echo "[3/5] Running rule-based baseline..."
python baseline/rule_based_agent.py
echo ""

# --- Step 4: Grader score variance check ---
# Runs 20 full random-agent episodes across all three tasks and checks that the
# composite grader produces sufficient variance (stdev > 0.05).
# Graders that always return the same score are broken — this catches that.
echo "[4/5] Checking grader score variance..."
python -c "
from graders.composite_grader import CompositeGrader
from asha_env.env import AshaEnv
import statistics
import random

env = AshaEnv()
grader = CompositeGrader()
scores = []

for _ in range(20):
    # Randomly pick a task so variance spans all difficulty levels.
    task = random.choice(['easy_diagnosis', 'medium_consultation', 'hard_complex_case'])
    obs = env.reset(task)
    done = False
    trajectory = []

    # Run a full random episode to completion.
    while not done:
        action = random.choice(obs['available_actions'])
        obs, reward, done, info = env.step(action)
        trajectory.append(action)

    state = env.get_state()
    score = grader.grade(trajectory, state['true_diagnosis'], state['patient'])
    scores.append(score)

mean = statistics.mean(scores)
stdev = statistics.stdev(scores)
print(f'  Mean:  {mean:.4f}')
print(f'  Stdev: {stdev:.4f}')
print(f'  Min:   {min(scores):.4f}')
print(f'  Max:   {max(scores):.4f}')

# Fail the pipeline if variance is too low — graders may be broken.
if stdev < 0.05:
    print('  FAIL: Stdev < 0.05 — graders return near-constant scores!')
    exit(1)
print('  PASS: Score variance is sufficient.')
"
echo ""

# --- Step 5: Docker build (optional) ---
# Only runs if Docker is installed. Verifies the Dockerfile builds the image
# correctly. Skipped gracefully on machines without Docker.
echo "[5/5] Checking Docker build..."
if command -v docker &> /dev/null; then
    docker build -t asha-env .
    echo "  Docker build succeeded."
    echo "  To test: docker run -p 7860:7860 asha-env"
else
    echo "  Docker not available — skipping build test."
    echo "  Make sure to test Docker build before deploying."
fi

echo ""
echo "============================================"
echo "All local checks passed!"
echo "============================================"
