"""
DiagnosisGrader — scores whether the agent correctly identified the disease.

This is the most heavily weighted sub-grader (40% of the composite score).

Scoring rubric:
    No diagnosis made, but referred        →  0.3  (implicit acknowledgement of severity)
    No diagnosis made, no referral,
      emergency disease                    →  0.0  (dangerous miss)
    No diagnosis made, no referral,
      non-emergency                        →  0.1  (incomplete episode)
    Correct diagnosis                      →  0.9 + speed_bonus (up to 1.0)
      speed_bonus = 0.1 * (1 - steps/20)  — rewards reaching the right answer quickly
    Wrong disease, same category           →  0.5  (e.g. confused two maternal conditions)
    Wrong disease, different category,
      but disease is emergency             →  0.0  (dangerous miss)
    Wrong disease otherwise                →  0.2
"""

import json
import os

from .base_grader import BaseGrader

# Path to the diseases.json data file relative to this file's location.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")


def _load_diseases() -> dict:
    """Load diseases.json and return a dict keyed by disease id."""
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


class DiagnosisGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Score the diagnosis decision in the trajectory.

        Looks for the first "diagnose:<disease_id>" action in the trajectory
        and compares it to the true diagnosis. If no diagnose action was taken,
        checks for a refer action to award partial credit.

        Args:
            trajectory:     Ordered list of action strings from the episode.
            true_diagnosis: Ground-truth disease_id for this episode.
            patient:        Full patient dict (unused here, kept for interface consistency).

        Returns:
            Float in [0.0, 1.0]. See module docstring for full rubric.
        """
        diseases_db = _load_diseases()
        true_disease = diseases_db.get(true_diagnosis, {})
        true_category = true_disease.get("category", "")
        is_emergency = true_disease.get("emergency", False)

        # Scan trajectory for the first diagnose: action (only one is ever taken
        # per episode since diagnose is terminal).
        diagnosed_id = None
        for action in trajectory:
            if action.startswith("diagnose:"):
                diagnosed_id = action.split(":", 1)[1]
                break

        # --- No diagnosis was made ---
        if diagnosed_id is None:
            # A referral without diagnosis still shows clinical awareness.
            for action in trajectory:
                if action.startswith("refer:"):
                    return 0.3
            # Completely missed an emergency with no referral either → most dangerous outcome.
            if is_emergency:
                return 0.0
            # Non-emergency left undiagnosed — episode just ended without a decision.
            return 0.1

        # --- Correct diagnosis ---
        if diagnosed_id == true_diagnosis:
            steps = len(trajectory)
            # Speed bonus: reaching the correct diagnosis in fewer steps earns up to +0.1.
            # At 0 steps → +0.1 bonus. At 20+ steps → no bonus.
            speed_bonus = max(0.0, 0.1 * (1.0 - steps / 20.0))
            return min(1.0, 0.9 + speed_bonus)

        # --- Wrong diagnosis — check how wrong ---
        diag_disease = diseases_db.get(diagnosed_id, {})
        # Same category (e.g. both "maternal") — shows general domain awareness.
        if diag_disease.get("category") == true_category:
            return 0.5

        # Wrong category AND the true disease was an emergency → patient at serious risk.
        if is_emergency:
            return 0.0

        # Wrong diagnosis, non-emergency — minor partial credit for attempting.
        return 0.2
