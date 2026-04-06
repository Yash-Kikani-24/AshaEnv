"""
Comorbidity logic for patient generation.

Handles which diseases can appear together as comorbidities, and which ones
are mutually exclusive. On medium/hard difficulty, the patient may have a
secondary condition alongside their primary disease, making diagnosis harder.
"""

import random

# Pairs of diseases that are mutually exclusive — a patient cannot have both
# at the same time (e.g., antepartum and postpartum haemorrhage).
CANNOT_COEXIST = [
    ("antepartum_haemorrhage", "postpartum_haemorrhage"),
    ("birth_asphyxia", "neonatal_jaundice"),
    ("eclampsia", "hyperemesis"),
]

# Maps a primary disease → list of diseases that commonly co-occur with it.
# E.g., severe_anaemia often leads to low_birth_weight or postpartum_haemorrhage.
COMORBID_PAIRS = {
    "severe_anaemia": ["low_birth_weight", "postpartum_haemorrhage"],
    "pre_eclampsia": ["severe_anaemia", "low_birth_weight"],
    "eclampsia": ["pre_eclampsia"],
    "gestational_diabetes": ["pre_eclampsia", "low_birth_weight"],
    "postpartum_haemorrhage": ["severe_anaemia"],
    "low_birth_weight": ["hypothermia_newborn", "neonatal_sepsis"],
    "hypothermia_newborn": ["low_birth_weight", "neonatal_sepsis"],
    "neonatal_sepsis": ["low_birth_weight"],
    "preterm_labour": ["low_birth_weight"],
    "obstructed_labour": ["birth_asphyxia", "postpartum_haemorrhage"],
}


def _can_coexist(disease_a: str, disease_b: str) -> bool:
    """Check whether two diseases are allowed to appear together (not in CANNOT_COEXIST)."""
    for a, b in CANNOT_COEXIST:
        if (disease_a == a and disease_b == b) or (disease_a == b and disease_b == a):
            return False
    return True


def get_comorbidities(
    primary_disease: str, task_difficulty: str
) -> list[str]:
    """
    Randomly select 0 or 1 comorbid diseases for the given primary disease.

    - Easy tasks: never have comorbidities.
    - Medium tasks: 10% chance of one comorbidity.
    - Hard tasks: 30% chance of one comorbidity.

    Returns a list of 0 or 1 disease ID strings.
    """
    if task_difficulty == "easy":
        return []

    comorbidity_chance = 0.1 if task_difficulty == "medium" else 0.30

    if random.random() > comorbidity_chance:
        return []

    candidates = COMORBID_PAIRS.get(primary_disease, [])
    valid = [c for c in candidates if _can_coexist(primary_disease, c)]

    if not valid:
        return []

    return [random.choice(valid)]
