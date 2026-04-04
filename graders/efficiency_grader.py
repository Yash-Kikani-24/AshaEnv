import json
import os

from .base_grader import BaseGrader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")


def _load_diseases():
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


def _load_symptoms():
    with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
        return json.load(f)


class EfficiencyGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        diseases_db = _load_diseases()
        symptoms_db = _load_symptoms()
        true_disease = diseases_db.get(true_diagnosis, {})

        # Count question actions
        questions_asked = []
        relevant_questions = []
        repeated_questions = set()
        seen_questions = set()

        relevant_symptoms = set(
            true_disease.get("required_symptoms", [])
            + true_disease.get("optional_symptoms", [])
        )

        for action in trajectory:
            if action.startswith("ask_symptom:") or action.startswith("ask_history:"):
                item = action.split(":", 1)[1]
                questions_asked.append(action)

                if item in seen_questions:
                    repeated_questions.add(item)
                seen_questions.add(item)

                if action.startswith("ask_symptom:") and item in relevant_symptoms:
                    relevant_questions.append(action)

        total_questions = len(questions_asked)
        if total_questions == 0:
            # No questions asked — could be good (easy case) or bad
            # Check if correct diagnosis was made
            for action in trajectory:
                if action.startswith("diagnose:"):
                    diag = action.split(":", 1)[1]
                    if diag == true_diagnosis:
                        return 1.0
                    return 0.3
            return 0.2

        # Minimum questions needed = number of required symptoms
        min_needed = len(true_disease.get("required_symptoms", []))
        min_needed = max(min_needed, 1)

        # Base efficiency: min_needed / actual
        efficiency = min(1.0, min_needed / total_questions)

        # Penalty for repeated questions
        repeat_penalty = len(repeated_questions) * 0.1

        # Penalty for irrelevant questions (scales with how many were irrelevant)
        irrelevant_count = total_questions - len(relevant_questions) - len(repeated_questions)
        irrelevant_ratio = max(0, irrelevant_count) / max(1, total_questions)
        irrelevant_penalty = irrelevant_ratio * 0.4

        # Bonus for reaching correct diagnosis
        diagnosis_bonus = 0.0
        for action in trajectory:
            if action.startswith("diagnose:"):
                if action.split(":", 1)[1] == true_diagnosis:
                    diagnosis_bonus = 0.2
                break

        score = efficiency - repeat_penalty - irrelevant_penalty + diagnosis_bonus
        return max(0.0, min(1.0, score))
