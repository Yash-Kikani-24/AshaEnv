"""Rule-based baseline agent with handcrafted clinical decision rules."""

import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader

TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]
EPISODES_PER_TASK = 10

# Priority order of symptoms to ask
TRIAGE_SYMPTOMS = [
    "fever", "cough", "diarrhoea", "chills", "headache",
    "rash", "jaundice", "shortness_of_breath", "high_bp",
    "weight_loss", "persistent_cough", "abdominal_pain",
    "swelling", "frequent_urination", "pallor",
]

# Decision rules: (required symptoms present) → (action to take)
RULES = [
    # Emergency: pre-eclampsia
    ({"high_bp", "swelling", "headache"}, "refer:district_hospital"),
    ({"high_bp", "swelling"}, "refer:district_hospital"),
    # Malaria
    ({"fever", "chills", "sweating"}, "order_test:malaria_rdt"),
    ({"fever", "chills"}, "order_test:malaria_rdt"),
    # TB
    ({"persistent_cough", "weight_loss", "night_sweats"}, "refer:district_hospital"),
    ({"persistent_cough", "weight_loss"}, "refer:phc"),
    # Pneumonia
    ({"fever", "productive_cough", "shortness_of_breath"}, "refer:phc"),
    ({"fever", "cough", "shortness_of_breath"}, "refer:phc"),
    # Dengue
    ({"fever", "headache", "joint_pain", "rash"}, "refer:district_hospital"),
    ({"fever", "headache", "joint_pain"}, "diagnose:dengue"),
    # Typhoid
    ({"fever", "abdominal_pain", "weakness"}, "diagnose:typhoid"),
    # Diarrhoea
    ({"diarrhoea", "dehydration"}, "diagnose:diarrhoea"),
    ({"diarrhoea", "abdominal_pain"}, "diagnose:diarrhoea"),
    # Anaemia
    ({"fatigue", "pallor", "weakness"}, "diagnose:anaemia"),
    ({"fatigue", "pallor"}, "order_test:haemoglobin_strip"),
    # Malnutrition
    ({"weight_loss", "weakness", "fatigue"}, "diagnose:malnutrition"),
    # Chickenpox
    ({"rash", "fever", "itching"}, "diagnose:chickenpox"),
    # Jaundice
    ({"jaundice", "fatigue"}, "refer:district_hospital"),
    # Worms
    ({"abdominal_pain", "loss_of_appetite", "itching_anus"}, "diagnose:worm_infestation"),
    # ARI
    ({"cough", "fever", "runny_nose"}, "diagnose:ari"),
    ({"cough", "runny_nose"}, "diagnose:ari"),
    # Hypertension
    ({"high_bp", "headache"}, "diagnose:hypertension"),
    # Diabetes
    ({"frequent_urination", "excessive_thirst"}, "diagnose:diabetes"),
]

# After diagnosis, treatment rules
TREATMENT_RULES = {
    "malaria": ["treat:chloroquine", "treat:paracetamol"],
    "diarrhoea": ["treat:ors", "treat:zinc"],
    "anaemia": ["treat:iron_tablets"],
    "malnutrition": ["treat:iron_tablets", "treat:zinc"],
    "ari": ["treat:paracetamol"],
    "worm_infestation": ["treat:albendazole"],
}


def pick_action(obs: dict, asked: set, diagnosed: str, test_results: set) -> str:
    available = obs.get("available_actions", [])
    if not available:
        return None

    known = set(obs["patient"].get("known_symptoms", []))

    # If already diagnosed, try treatment
    if diagnosed:
        treatments = TREATMENT_RULES.get(diagnosed, [])
        for t in treatments:
            if t in available:
                return t
        # If no treatment available, refer
        for a in available:
            if a.startswith("refer:"):
                return a

    # After malaria RDT test, decide
    if "malaria_rdt" in test_results and "fever" in known and "chills" in known:
        if "diagnose:malaria" in available:
            return "diagnose:malaria"

    # After haemoglobin test, decide
    if "haemoglobin_strip" in test_results and "pallor" in known:
        if "diagnose:anaemia" in available:
            return "diagnose:anaemia"

    # Check rules against known symptoms
    for required_symptoms, action in RULES:
        if required_symptoms.issubset(known):
            if action in available:
                return action

    # Ask next triage symptom
    for symptom in TRIAGE_SYMPTOMS:
        action = f"ask_symptom:{symptom}"
        if action in available and symptom not in asked:
            return action

    # If we've asked enough, try to diagnose based on what we know
    if len(asked) >= 5:
        # Pick the most likely diagnosis action
        for action in available:
            if action.startswith("diagnose:"):
                return action

    # Fallback: first available ask
    for action in available:
        if action.startswith("ask_symptom:"):
            return action

    return available[0]


def run_rule_episode(env: AshaEnv, grader: CompositeGrader, task_id: str) -> float:
    obs = env.reset(task_id)
    done = False
    trajectory = []
    asked = set()
    diagnosed = None
    test_results = set()

    while not done:
        action = pick_action(obs, asked, diagnosed, test_results)
        if action is None:
            break

        if action.startswith("ask_symptom:") or action.startswith("ask_history:"):
            asked.add(action.split(":", 1)[1])
        elif action.startswith("diagnose:"):
            diagnosed = action.split(":", 1)[1]
        elif action.startswith("order_test:"):
            test_results.add(action.split(":", 1)[1])

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
            score = run_rule_episode(env, grader, task_id)
            scores.append(score)
        results[task_id] = scores

    # Print results
    print(f"\n{'='*60}")
    print("Rule-Based Agent Baseline")
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
