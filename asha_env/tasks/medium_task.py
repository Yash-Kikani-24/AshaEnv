"""
Medium task configuration — realistic ASHA consultation scenario.

- 10 diseases including some critical ones (pre-eclampsia, neonatal sepsis).
- Only the chief complaint is revealed; agent must ask about other symptoms.
- 10% chance of a comorbidity (secondary condition).
- ±5% noise on vital signs (simulates basic instrument imprecision).
- 10% patient non-compliance rate (may give unreliable answers).
- Agent has 12 steps → must balance information gathering vs. decision-making.
"""


class MediumTask:
    task_id = "medium_consultation"
    name = "Medium Consultation"
    diseases = [
        "severe_anaemia", "neonatal_jaundice", "low_birth_weight",
        "hyperemesis", "hypothermia_newborn", "pre_eclampsia",
        "postpartum_haemorrhage", "neonatal_sepsis", "gestational_diabetes",
        "puerperal_sepsis",
    ]
    max_steps = 12
    difficulty = "medium"
    all_symptoms_revealed = False
    comorbidity_chance = 0.1
    vitals_noise = 0.05
