import json
import os

from .base_grader import BaseGrader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")

REFERRAL_LEVELS = ["none", "phc", "district_hospital", "emergency"]


def _load_diseases():
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


class ReferralGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        diseases_db = _load_diseases()
        true_disease = diseases_db.get(true_diagnosis, {})
        correct_level = true_disease.get("referral_level", "none")
        is_emergency = true_disease.get("emergency", False)
        treatable = true_disease.get("treatable_with_kit", False)

        # Find referral action
        refer_level = None
        diagnosed = False
        treated = False

        for action in trajectory:
            if action.startswith("refer:"):
                refer_level = action.split(":", 1)[1]
            elif action.startswith("diagnose:"):
                diagnosed = True
            elif action.startswith("treat:"):
                treated = True

        # Case 1: No referral made
        if refer_level is None:
            if correct_level == "none" and treatable:
                # Correct — disease can be handled at village level
                if treated:
                    return 1.0
                elif diagnosed:
                    return 0.6
                return 0.4
            elif is_emergency:
                # Missed emergency referral
                return 0.0
            else:
                # Should have referred but didn't
                return 0.2

        # Case 2: Referral made
        if refer_level not in REFERRAL_LEVELS:
            return 0.1

        given_idx = REFERRAL_LEVELS.index(refer_level)
        correct_idx = REFERRAL_LEVELS.index(correct_level) if correct_level in REFERRAL_LEVELS else 1

        if refer_level == correct_level:
            return 1.0
        elif given_idx > correct_idx:
            # Over-referred — wastes resources but patient is safe
            return 0.5
        else:
            # Under-referred
            if is_emergency:
                return 0.0
            return max(0.0, 1.0 - (correct_idx - given_idx) * 0.35)
