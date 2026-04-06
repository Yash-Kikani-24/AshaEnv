"""
ReferralGrader — scores whether the agent sent the patient to the right facility.

This grader (15% of composite score) evaluates the referral decision independently
from the diagnosis. The four referral levels (ordered lowest to highest):

    none              — patient can be managed at village level (no referral needed)
    phc               — Primary Health Centre
    district_hospital — district-level facility with surgical/specialist capacity
    emergency         — immediate emergency transport

Scoring rubric:

  No referral made:
    correct_level == "none" AND treatable_with_kit:
      agent treated the patient                 → 1.0  (perfect village-level management)
      agent diagnosed but did not treat         → 0.6  (partial — knew the disease, missed treatment)
      neither diagnosed nor treated             → 0.4  (got lucky on the referral decision)
    is_emergency AND no referral               → 0.0  (critical miss)
    should have referred but didn't            → 0.2

  Referral made:
    unknown level (not in REFERRAL_LEVELS)     → 0.1  (invalid action)
    exact correct level                        → 1.0
    over-referred (higher than needed)         → 0.5  (safe but wastes resources)
    under-referred, emergency disease          → 0.0  (dangerous)
    under-referred, non-emergency              → 1.0 − gap × 0.35  (e.g. 1 level off → 0.65)
"""

import json
import os

from .base_grader import BaseGrader

# Path to the data directory relative to this file's location.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "asha_env", "data")

# Ordered from least to most urgent. Index position is used for gap arithmetic
# (e.g. district_hospital is index 2; phc is index 1; gap = 1).
REFERRAL_LEVELS = ["none", "phc", "district_hospital", "emergency"]


def _load_diseases() -> dict:
    """Load diseases.json and return a dict keyed by disease id."""
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


class ReferralGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Score the agent's referral decision for this episode.

        Scans the trajectory for the last refer: action and compares it to the
        disease's required referral level. Also checks for diagnosis and treatment
        actions to handle the "no referral needed" case correctly.

        Args:
            trajectory:     Ordered list of action strings from the episode.
            true_diagnosis: Ground-truth disease_id for this episode.
            patient:        Full patient dict (unused here, kept for interface consistency).

        Returns:
            Float in [0.0, 1.0]. See module docstring for full rubric.
        """
        diseases_db = _load_diseases()
        true_disease = diseases_db.get(true_diagnosis, {})
        correct_level = true_disease.get("referral_level", "none")
        is_emergency = true_disease.get("emergency", False)
        # treatable_with_kit = True means the ASHA can manage this without referral.
        treatable = true_disease.get("treatable_with_kit", False)

        # Scan trajectory to find the referral level (if any) and whether a
        # diagnosis or treatment was attempted.
        refer_level = None
        diagnosed = False
        treated = False

        for action in trajectory:
            if action.startswith("refer:"):
                # Take the last refer: action in case the agent issued more than one.
                refer_level = action.split(":", 1)[1]
            elif action.startswith("diagnose:"):
                diagnosed = True
            elif action.startswith("treat:"):
                treated = True

        # --- Case 1: No referral was made ---
        if refer_level is None:
            if correct_level == "none" and treatable:
                # Disease is manageable at village level — no referral is correct.
                # Award based on how complete the agent's village management was.
                if treated:
                    return 1.0   # Diagnosed, treated, no unnecessary referral — perfect.
                elif diagnosed:
                    return 0.6   # Identified but forgot to treat.
                return 0.4       # Neither — but at least didn't refer unnecessarily.
            elif is_emergency:
                # Agent completely failed to escalate a life-threatening case.
                return 0.0
            else:
                # Non-emergency disease that needed referral, but agent didn't refer.
                return 0.2

        # --- Case 2: A referral was made ---
        if refer_level not in REFERRAL_LEVELS:
            # Agent produced an unrecognised referral level — invalid action.
            return 0.1

        given_idx = REFERRAL_LEVELS.index(refer_level)
        # Default to PHC (index 1) if the disease has an unexpected referral_level value.
        correct_idx = REFERRAL_LEVELS.index(correct_level) if correct_level in REFERRAL_LEVELS else 1

        if refer_level == correct_level:
            return 1.0  # Exact match — best outcome.
        elif given_idx > correct_idx:
            # Over-referred — patient will be seen at a higher facility than needed.
            # Safe but wastes scarce referral capacity.
            return 0.5
        else:
            # Under-referred — patient sent to a facility that may lack needed resources.
            if is_emergency:
                # Under-referring an emergency is extremely dangerous.
                return 0.0
            # For non-emergencies, apply a graded penalty based on how many levels off.
            # 1 level off → 0.65, 2 levels off → 0.30, 3 levels off → 0.0
            return max(0.0, 1.0 - (correct_idx - given_idx) * 0.35)
