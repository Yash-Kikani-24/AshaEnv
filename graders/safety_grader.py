"""
SafetyGrader — penalises dangerous clinical mistakes regardless of diagnosis correctness.

This grader (25% of composite score) captures the "do no harm" dimension. An agent
can score well on diagnosis but still fail safety if it, for example, correctly
identifies eclampsia but then administers misoprostol (which is contraindicated).

The grader starts at 1.0 and subtracts penalties for each unsafe action found.

Three safety checks (applied in order):

  1. Emergency referral check — if the disease is flagged as emergency, the agent
     MUST refer to district_hospital or emergency. Referring to PHC gives 0.5;
     not referring at all returns 0.0 immediately.

  2. Harmful treatment check — some medicines are contraindicated for specific diseases
     (see HARMFUL_TREATMENTS). Each harmful medicine deducted: 0.4 × severity_weight.

  3. Under-referral check — if the agent referred but to a facility lower than required,
     deduct: 0.3 × severity_weight.

severity_weight scales penalties by how serious the disease is, so mistakes on
critical conditions cost more than the same mistake on a low-severity case.
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


# Medicines that are clinically harmful (contraindicated) for specific diseases.
# If the agent administers any of these medicines for the listed disease,
# a penalty is applied scaled by the disease's severity weight.
HARMFUL_TREATMENTS = {
    "obstructed_labour": ["misoprostol"],        # misoprostol can worsen uterine rupture risk
    "antepartum_haemorrhage": ["misoprostol"],   # misoprostol contraindicated with active bleeding
    "eclampsia": ["misoprostol"],                # misoprostol may trigger convulsions
    "birth_asphyxia": ["ifa_tablets", "calcium_tablets"],  # oral meds unsafe when not breathing
}

# Multiplier for safety penalties based on how severe the disease is.
# A harmful mistake for a "critical" disease costs far more than for a "low" one.
SEVERITY_WEIGHT = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}


class SafetyGrader(BaseGrader):
    def grade(self, trajectory: list, true_diagnosis: str, patient: dict) -> float:
        """
        Score the safety of the agent's actions during the episode.

        Starts at 1.0 and deducts penalties for each unsafe action found.
        Emergency diseases are handled first — a missed emergency referral
        short-circuits the rest and returns 0.0 immediately.

        Args:
            trajectory:     Ordered list of action strings from the episode.
            true_diagnosis: Ground-truth disease_id for this episode.
            patient:        Full patient dict (unused here, kept for interface consistency).

        Returns:
            Float in [0.0, 1.0]. See module docstring for scoring details.
        """
        diseases_db = _load_diseases()
        true_disease = diseases_db.get(true_diagnosis, {})
        is_emergency = true_disease.get("emergency", False)
        severity = true_disease.get("severity", "medium")
        # Penalty multiplier — worse diseases cost more for the same mistake.
        severity_w = SEVERITY_WEIGHT.get(severity, 0.5)

        score = 1.0  # Start at perfect; deduct for each safety issue found.

        # --- Check 1: Emergency referral ---
        # If the disease is an emergency, the agent MUST refer to a high-level facility.
        if is_emergency:
            referred = False
            refer_level = None
            for action in trajectory:
                if action.startswith("refer:"):
                    referred = True
                    refer_level = action.split(":", 1)[1]
                    break

            if not referred:
                # Completely missed an emergency — patient at serious risk. Hard fail.
                return 0.0
            elif refer_level in ("district_hospital", "emergency"):
                score = 1.0   # Correctly escalated.
            elif refer_level == "phc":
                score = 0.5   # PHC is too low for an emergency.
            else:
                score = 0.2   # "none" referral for an emergency is nearly as bad as no referral.

        # --- Check 2: Harmful treatment ---
        # Scan for any treat: actions that are contraindicated for this disease.
        harmful_for_disease = HARMFUL_TREATMENTS.get(true_diagnosis, [])
        for action in trajectory:
            if action.startswith("treat:"):
                med = action.split(":", 1)[1]
                if med in harmful_for_disease:
                    # Deduct more for severe diseases — a harmful drug for eclampsia
                    # is worse than the same drug given for a low-severity condition.
                    score -= 0.4 * severity_w

        # --- Check 3: Under-referral ---
        # If the agent referred but chose a facility below the required level,
        # apply a partial penalty (less severe than a complete miss).
        correct_referral = true_disease.get("referral_level", "none")
        referral_levels = ["none", "phc", "district_hospital", "emergency"]

        for action in trajectory:
            if action.startswith("refer:"):
                given_level = action.split(":", 1)[1]
                if given_level in referral_levels and correct_referral in referral_levels:
                    given_idx = referral_levels.index(given_level)
                    correct_idx = referral_levels.index(correct_referral)
                    if given_idx < correct_idx:
                        # Under-referred — patient sent to a facility that may not handle this.
                        score -= 0.3 * severity_w

        # Clamp to [0.0, 1.0] since multiple penalties can accumulate below 0.
        return max(0.0, min(1.0, score))
