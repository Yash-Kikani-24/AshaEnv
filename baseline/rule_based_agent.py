"""
Rule-based baseline agent for the ASHA environment.

This agent uses handcrafted clinical decision rules derived from standard ASHA
protocols for maternal and newborn health in rural India. It does NOT learn from
data — all logic is hard-coded by a domain expert.

Purpose:
    Establish a strong, interpretable upper bound. A well-trained ML/LLM agent
    should aim to match or exceed this baseline. If your agent scores significantly
    below it, your model is likely not using the symptom information effectively.

Decision logic (in priority order inside pick_action):
    1. Post-diagnosis treatment  — if a diagnosis was already made, give the right medicine.
    2. Test-result interpretation — after a lab test, convert the result into a diagnosis.
    3. Pattern matching rules     — if known symptoms match a clinical rule, act on it.
    4. Triage screening           — ask the highest-priority symptom not yet asked.
    5. Forced diagnosis fallback  — if ≥5 symptoms asked with no rule hit, guess.
    6. Last-resort fallback       — pick the first available action.

Usage:
    python baseline/rule_based_agent.py
"""

import statistics
import sys
import os

# Allow importing asha_env and graders from the project root when running this
# script directly (i.e., python baseline/rule_based_agent.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asha_env.env import AshaEnv
from graders.composite_grader import CompositeGrader

# The three task difficulties defined in AshaEnv.
TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]

# How many independent episodes to run per task for the benchmark summary.
EPISODES_PER_TASK = 10

# Ordered list of symptoms to ask during triage, from most to least clinically urgent.
# The agent works through this list in order, skipping symptoms already asked,
# until a decision rule fires or the list is exhausted.
TRIAGE_SYMPTOMS = [
    "vaginal_bleeding", "convulsions", "high_bp", "fever",
    "not_breathing", "not_crying", "severe_headache",
    "swelling_face_hands", "foul_smelling_discharge",
    "pallor", "excessive_vomiting", "yellow_skin",
    "poor_feeding", "hypothermia_sign", "low_weight",
    "contractions_before_37wks", "prolonged_labour",
    "fever_newborn", "lethargy_newborn", "weakness",
]

# Clinical decision rules: (frozenset of required symptoms) -> action string.
# Rules are checked in order — first match wins. Each rule represents a
# well-established ASHA protocol guideline for the listed condition.
# Action format mirrors env.step() action strings (e.g. "refer:district_hospital").
RULES = [
    # Emergency: eclampsia — convulsions with high BP or unconsciousness → immediate referral.
    ({"convulsions", "high_bp"}, "refer:district_hospital"),
    ({"convulsions", "unconsciousness"}, "refer:district_hospital"),
    # Emergency: birth asphyxia — not breathing or blue at birth → immediate referral.
    ({"not_breathing", "not_crying"}, "refer:district_hospital"),
    ({"not_breathing", "blue_skin"}, "refer:district_hospital"),
    # Emergency: antepartum haemorrhage — bleeding + pain → immediate referral.
    ({"vaginal_bleeding", "abdominal_pain"}, "refer:district_hospital"),
    # Pre-eclampsia — high BP + swelling + severe headache → refer.
    ({"high_bp", "swelling_face_hands", "severe_headache"}, "refer:district_hospital"),
    ({"high_bp", "swelling_face_hands"}, "refer:district_hospital"),
    # Postpartum haemorrhage — post-delivery bleeding with weakness/rapid heartbeat.
    ({"vaginal_bleeding", "weakness", "rapid_heartbeat"}, "diagnose:postpartum_haemorrhage"),
    ({"vaginal_bleeding", "weakness"}, "diagnose:postpartum_haemorrhage"),
    # Obstructed labour — prolonged labour with distress → refer.
    ({"prolonged_labour", "severe_abdominal_pain"}, "refer:district_hospital"),
    ({"prolonged_labour", "maternal_distress"}, "refer:district_hospital"),
    # Preterm labour — contractions before 37 weeks → refer.
    ({"contractions_before_37wks", "lower_back_pain"}, "refer:district_hospital"),
    ({"contractions_before_37wks", "pelvic_pressure"}, "refer:district_hospital"),
    # Puerperal sepsis — fever + foul-smelling discharge after delivery.
    ({"fever", "foul_smelling_discharge"}, "diagnose:puerperal_sepsis"),
    ({"fever", "foul_smelling_discharge", "lower_abdominal_pain"}, "diagnose:puerperal_sepsis"),
    # Severe anaemia — pallor + fatigue/weakness → diagnose or order haemoglobin test.
    ({"pallor", "fatigue", "weakness"}, "diagnose:severe_anaemia"),
    ({"pallor", "weakness"}, "order_test:haemoglobin_strip"),
    # Neonatal sepsis — fever + poor feeding in newborn.
    ({"fever_newborn", "poor_feeding", "lethargy_newborn"}, "refer:district_hospital"),
    ({"fever_newborn", "poor_feeding"}, "diagnose:neonatal_sepsis"),
    # Neonatal jaundice — yellow skin + yellow eyes.
    ({"yellow_skin", "yellow_eyes"}, "diagnose:neonatal_jaundice"),
    # Low birth weight — low weight with weak cry or poor feeding.
    ({"low_weight", "weak_cry"}, "diagnose:low_birth_weight"),
    ({"low_weight", "poor_feeding"}, "diagnose:low_birth_weight"),
    # Hypothermia in newborn — cold signs + lethargy.
    ({"hypothermia_sign", "cold_extremities"}, "diagnose:hypothermia_newborn"),
    ({"hypothermia_sign", "lethargy_newborn"}, "diagnose:hypothermia_newborn"),
    # Hyperemesis — excessive vomiting leading to dehydration or weight loss.
    ({"excessive_vomiting", "dehydration"}, "diagnose:hyperemesis"),
    ({"excessive_vomiting", "weight_loss"}, "diagnose:hyperemesis"),
    # Gestational diabetes — frequent urination + thirst → diagnose or order dipstick test.
    ({"frequent_urination", "excessive_thirst", "fatigue"}, "diagnose:gestational_diabetes"),
    ({"frequent_urination", "excessive_thirst"}, "order_test:urine_dipstick"),
]

# After a diagnosis is made, map disease_id → list of treatment actions to attempt.
# These map to "treat:<medicine>" actions available in the ASHA kit.
TREATMENT_RULES = {
    "severe_anaemia": ["treat:ifa_tablets"],
    "postpartum_haemorrhage": ["treat:misoprostol"],
    "hyperemesis": ["treat:ors"],
    "puerperal_sepsis": ["treat:paracetamol"],
    "low_birth_weight": ["treat:ors"],
}


def pick_action(obs: dict, asked: set, diagnosed: str, test_results: set) -> str:
    """
    Choose the next action using the clinical rule priority ladder.

    This function is the core of the rule-based agent. It evaluates the current
    observation against a prioritised decision ladder and returns the best action.

    Priority order:
        1. If a diagnosis was already made → give the mapped treatment, or refer if
           no treatment is in the kit.
        2. If a haemoglobin test result is available and pallor was found → diagnose
           severe anaemia (test-result interpretation).
        3. If a urine dipstick result is available and frequent urination was found →
           diagnose gestational diabetes.
        4. Match known symptoms against RULES (first matching rule wins).
        5. Walk through TRIAGE_SYMPTOMS in order and ask the next unasked symptom.
        6. If ≥5 symptoms asked with no rule fire → pick the first diagnose: action
           (best guess under uncertainty).
        7. Last resort → first available ask_symptom: action, then available[0].

    Args:
        obs:          Current observation dict from env.step() or env.reset().
        asked:        Set of symptom/history IDs already asked this episode.
        diagnosed:    The disease_id string if a diagnose: action was already taken,
                      else None.
        test_results: Set of test IDs (e.g. "haemoglobin_strip") ordered so far.

    Returns:
        An action string compatible with env.step(), or None if no actions are left.
    """
    available = obs.get("available_actions", [])
    if not available:
        return None

    # Symptoms the patient has confirmed so far (revealed during the episode).
    known = set(obs["patient"].get("known_symptoms", []))

    # --- Priority 1: Post-diagnosis treatment ---
    if diagnosed:
        treatments = TREATMENT_RULES.get(diagnosed, [])
        for t in treatments:
            if t in available:
                return t
        # No applicable treatment in kit → refer the patient.
        for a in available:
            if a.startswith("refer:"):
                return a

    # --- Priority 2: Interpret haemoglobin test result ---
    # The haemoglobin strip was ordered and pallor is confirmed → severe anaemia.
    if "haemoglobin_strip" in test_results and "pallor" in known:
        if "diagnose:severe_anaemia" in available:
            return "diagnose:severe_anaemia"

    # --- Priority 3: Interpret urine dipstick result ---
    # Dipstick ordered and frequent urination confirmed → gestational diabetes.
    if "urine_dipstick" in test_results and "frequent_urination" in known:
        if "diagnose:gestational_diabetes" in available:
            return "diagnose:gestational_diabetes"

    # --- Priority 4: Pattern-matching clinical rules ---
    # Evaluate each rule; fire the first one whose symptom set is a subset of
    # what the agent currently knows about the patient.
    for required_symptoms, action in RULES:
        if required_symptoms.issubset(known):
            if action in available:
                return action

    # --- Priority 5: Systematic triage screening ---
    # Walk through the high-priority symptom list in order and ask each one
    # that hasn't been asked yet.
    for symptom in TRIAGE_SYMPTOMS:
        action = f"ask_symptom:{symptom}"
        if action in available and symptom not in asked:
            return action

    # --- Priority 6: Forced diagnosis after sufficient questioning ---
    # If ≥5 symptoms have been asked and no rule has fired, the case is ambiguous.
    # Make a best-guess diagnosis rather than continuing to ask indefinitely.
    if len(asked) >= 5:
        for action in available:
            if action.startswith("diagnose:"):
                return action

    # --- Priority 7: Last-resort fallbacks ---
    for action in available:
        if action.startswith("ask_symptom:"):
            return action

    return available[0]


def run_rule_episode(env: AshaEnv, grader: CompositeGrader, task_id: str) -> float:
    """
    Run one episode with the rule-based agent and return the composite score.

    Maintains episode-level state (which symptoms were asked, whether a diagnosis
    was made, which tests were ordered) and passes it into pick_action each step.

    Args:
        env:     The AshaEnv instance (reused across episodes).
        grader:  CompositeGrader that scores the full trajectory after the episode.
        task_id: One of TASKS — controls difficulty and allowed diseases.

    Returns:
        A float in [0.0, 1.0] representing the composite grader score.
    """
    obs = env.reset(task_id)
    done = False
    trajectory = []      # All actions taken (needed by grader).
    asked = set()        # Symptom/history IDs asked so far (avoids re-asking).
    diagnosed = None     # Set once a diagnose: action is taken.
    test_results = set() # Tracks which tests have been ordered.

    while not done:
        action = pick_action(obs, asked, diagnosed, test_results)
        if action is None:
            break

        # Update local state so pick_action has current context next iteration.
        if action.startswith("ask_symptom:") or action.startswith("ask_history:"):
            asked.add(action.split(":", 1)[1])
        elif action.startswith("diagnose:"):
            diagnosed = action.split(":", 1)[1]
        elif action.startswith("order_test:"):
            test_results.add(action.split(":", 1)[1])

        obs, reward, done, info = env.step(action)
        trajectory.append(action)

    # Retrieve ground-truth so the grader can score the trajectory.
    state = env.get_state()
    return grader.grade(trajectory, state["true_diagnosis"], state["patient"])


def main():
    """
    Benchmark the rule-based agent across all task difficulties and print a summary table.

    Runs EPISODES_PER_TASK independent episodes per task. Prints per-task and
    overall aggregate statistics (mean, min, max, stdev) to stdout.
    """
    env = AshaEnv()
    grader = CompositeGrader()

    # results[task_id] = list of float scores, one per episode.
    results = {}

    for task_id in TASKS:
        scores = []
        for _ in range(EPISODES_PER_TASK):
            score = run_rule_episode(env, grader, task_id)
            scores.append(score)
        results[task_id] = scores

    # --- Print results table ---
    print(f"\n{'='*60}")
    print("Rule-Based Agent Baseline (Maternal/Newborn)")
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
