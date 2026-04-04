import random

# Pairs of diseases that cannot coexist simultaneously
CANNOT_COEXIST = [
    ("malaria", "dengue"),
    ("chickenpox", "pre_eclampsia"),
    ("worm_infestation", "pre_eclampsia"),
]

# Diseases that commonly co-occur
COMORBID_PAIRS = {
    "anaemia": ["malnutrition", "worm_infestation"],
    "malnutrition": ["anaemia", "worm_infestation"],
    "hypertension": ["diabetes"],
    "diabetes": ["hypertension"],
    "diarrhoea": ["malnutrition"],
    "malaria": ["anaemia"],
    "tuberculosis": ["malnutrition", "anaemia"],
    "pre_eclampsia": ["hypertension", "anaemia"],
}


def _can_coexist(disease_a: str, disease_b: str) -> bool:
    for a, b in CANNOT_COEXIST:
        if (disease_a == a and disease_b == b) or (disease_a == b and disease_b == a):
            return False
    return True


def get_comorbidities(
    primary_disease: str, task_difficulty: str
) -> list[str]:
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
