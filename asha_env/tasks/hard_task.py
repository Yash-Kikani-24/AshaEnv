"""
Hard task configuration — complex emergency cases with all 15 diseases.

- Full disease pool including life-threatening emergencies (eclampsia, haemorrhages).
- Only the chief complaint is revealed; agent must actively investigate.
- 30% chance of comorbidity (overlapping symptoms make diagnosis harder).
- 20% patient non-compliance rate (unreliable answers, especially in emergencies).
- ±5% noise on vital signs.
- Agent has 20 steps but must identify emergencies quickly to avoid -1.0 penalty.
"""


class HardTask:
    task_id = "hard_complex_case"
    name = "Hard Complex Case"
    diseases = [
        "severe_anaemia", "pre_eclampsia", "eclampsia",
        "antepartum_haemorrhage", "postpartum_haemorrhage", "puerperal_sepsis",
        "gestational_diabetes", "hyperemesis", "preterm_labour",
        "obstructed_labour", "birth_asphyxia", "neonatal_sepsis",
        "neonatal_jaundice", "low_birth_weight", "hypothermia_newborn",
    ]
    max_steps = 20
    difficulty = "hard"
    all_symptoms_revealed = False
    comorbidity_chance = 0.30
    non_compliance_rate = 0.20
    emergency_cases = True
    vitals_noise = 0.05
