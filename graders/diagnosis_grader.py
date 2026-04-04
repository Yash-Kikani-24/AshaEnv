import json
import os

from .base_grader import BaseGrader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")


def _load_diseases():
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


class DiagnosisGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        diseases_db = _load_diseases()
        true_disease = diseases_db.get(true_diagnosis, {})
        true_category = true_disease.get("category", "")
        is_emergency = true_disease.get("emergency", False)

        # Find the diagnosis action in the trajectory
        diagnosed_id = None
        for action in trajectory:
            if action.startswith("diagnose:"):
                diagnosed_id = action.split(":", 1)[1]
                break

        # No diagnosis made
        if diagnosed_id is None:
            # Check if they referred instead — partial credit
            for action in trajectory:
                if action.startswith("refer:"):
                    return 0.3
            if is_emergency:
                return 0.0
            return 0.1

        # Correct diagnosis — bonus for fewer steps
        if diagnosed_id == true_diagnosis:
            steps = len(trajectory)
            speed_bonus = max(0.0, 0.1 * (1.0 - steps / 20.0))
            return min(1.0, 0.9 + speed_bonus)

        # Same category
        diag_disease = diseases_db.get(diagnosed_id, {})
        if diag_disease.get("category") == true_category:
            return 0.5

        # Missed emergency
        if is_emergency:
            return 0.0

        return 0.2
