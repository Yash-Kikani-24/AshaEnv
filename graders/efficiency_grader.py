"""
EfficiencyGrader — scores how efficiently the agent gathered information.

This grader (20% of composite score) rewards focused, purposeful questioning.
A real ASHA worker has limited time with each patient; asking many irrelevant
or repeated questions wastes precious consultation time.

Scoring formula:
    base_efficiency    = min(1.0, required_symptoms_count / total_questions_asked)
    repeat_penalty     = repeated_question count × 0.1
    irrelevant_penalty = (irrelevant_questions / total_questions) × 0.4
    diagnosis_bonus    = 0.2 if the correct diagnosis was made, else 0.0

    score = base_efficiency − repeat_penalty − irrelevant_penalty + diagnosis_bonus
    score clamped to [0.0, 1.0]

Key definitions:
    relevant question  — asks about a symptom listed in the disease's required_symptoms
                         or optional_symptoms fields in diseases.json.
    irrelevant question — any ask_symptom or ask_history not relevant to the true disease.
    repeated question   — asking about the same symptom/history item more than once.

Special case (zero questions asked):
    If the agent skipped all questions and still got the diagnosis right → 1.0 (perfect).
    If it guessed wrong → 0.3.
    If it never diagnosed at all → 0.2.
"""

import json
import os

from .base_grader import BaseGrader

# Path to the data directory relative to this file's location.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")


def _load_diseases() -> dict:
    """Load diseases.json and return a dict keyed by disease id."""
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


def _load_symptoms() -> dict:
    """Load symptoms.json and return the full symptoms dict."""
    with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
        return json.load(f)


class EfficiencyGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Score how efficiently the agent gathered information to reach its decision.

        Scans the trajectory for ask_symptom and ask_history actions, then measures
        how many were relevant, repeated, or wasted. A correct diagnosis at the end
        adds a bonus. See module docstring for the full formula.

        Args:
            trajectory:     Ordered list of action strings from the episode.
            true_diagnosis: Ground-truth disease_id for this episode.
            patient:        Full patient dict (unused here, kept for interface consistency).

        Returns:
            Float in [0.0, 1.0]. See module docstring for scoring details.
        """
        diseases_db = _load_diseases()
        symptoms_db = _load_symptoms()
        true_disease = diseases_db.get(true_diagnosis, {})

        # Collect all question actions from the trajectory.
        questions_asked = []      # All ask_symptom / ask_history actions (with duplicates).
        relevant_questions = []   # Subset of ask_symptom actions relevant to the true disease.
        repeated_questions = set()# Symptom/history IDs asked more than once.
        seen_questions = set()    # Tracks what's been asked to detect repeats.

        # Symptoms and history items that are clinically relevant for the true disease.
        relevant_symptoms = set(
            true_disease.get("required_symptoms", [])
            + true_disease.get("optional_symptoms", [])
        )

        for action in trajectory:
            if action.startswith("ask_symptom:") or action.startswith("ask_history:"):
                item = action.split(":", 1)[1]
                questions_asked.append(action)

                # Detect repeated questions — asking twice wastes a step.
                if item in seen_questions:
                    repeated_questions.add(item)
                seen_questions.add(item)

                # Track whether this symptom question was clinically relevant.
                if action.startswith("ask_symptom:") and item in relevant_symptoms:
                    relevant_questions.append(action)

        total_questions = len(questions_asked)

        # --- Special case: no questions asked ---
        if total_questions == 0:
            # Agent skipped questioning entirely. Score based on whether it was right.
            for action in trajectory:
                if action.startswith("diagnose:"):
                    diag = action.split(":", 1)[1]
                    if diag == true_diagnosis:
                        return 1.0   # Lucky/fast correct guess — maximum efficiency.
                    return 0.3       # Wrong without asking anything.
            return 0.2  # No diagnosis and no questions — incomplete episode.

        # --- Base efficiency: how close to the minimum was the agent? ---
        # Minimum questions needed is at least the number of required symptoms.
        min_needed = len(true_disease.get("required_symptoms", []))
        min_needed = max(min_needed, 1)  # Guard against diseases with no required_symptoms.

        # Efficiency ratio: if agent asked exactly the minimum → 1.0;
        # more questions → proportionally lower.
        efficiency = min(1.0, min_needed / total_questions)

        # --- Penalty: repeated questions ---
        # Each repeated symptom costs 0.1 regardless of severity.
        repeat_penalty = len(repeated_questions) * 0.1

        # --- Penalty: irrelevant questions ---
        # Count questions that were neither relevant nor repeated.
        irrelevant_count = total_questions - len(relevant_questions) - len(repeated_questions)
        # Normalise to a 0–1 ratio, then scale penalty up to 0.4.
        irrelevant_ratio = max(0, irrelevant_count) / max(1, total_questions)
        irrelevant_penalty = irrelevant_ratio * 0.4

        # --- Bonus: correct diagnosis ---
        # Add 0.2 if the agent ultimately reached the right conclusion.
        # This rewards efficiency + correctness together.
        diagnosis_bonus = 0.0
        for action in trajectory:
            if action.startswith("diagnose:"):
                if action.split(":", 1)[1] == true_diagnosis:
                    diagnosis_bonus = 0.2
                break  # Only one diagnose action per episode.

        score = efficiency - repeat_penalty - irrelevant_penalty + diagnosis_bonus
        return max(0.0, min(1.0, score))
