#!/bin/bash
set -e

echo "============================================"
echo "ASHA Agent Environment — Local Validation"
echo "============================================"
echo ""

# 1. Run tests
echo "[1/5] Running tests..."
python -m pytest tests/ -v
echo ""

# 2. Run random baseline
echo "[2/5] Running random baseline..."
python baseline/random_agent.py
echo ""

# 3. Run rule-based baseline
echo "[3/5] Running rule-based baseline..."
python baseline/rule_based_agent.py
echo ""

# 4. Score variance check
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
    task = random.choice(['easy_diagnosis', 'medium_consultation', 'hard_complex_case'])
    obs = env.reset(task)
    done = False
    trajectory = []
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
if stdev < 0.05:
    print('  FAIL: Stdev < 0.05 — graders return near-constant scores!')
    exit(1)
print('  PASS: Score variance is sufficient.')
"
echo ""

# 5. Docker build test (if docker available)
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
