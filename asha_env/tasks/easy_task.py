"""
Easy task configuration — straightforward single-disease diagnosis.

- Only 5 simple diseases (clear-cut symptoms, no emergencies).
- All symptoms are revealed upfront (no need to ask).
- No comorbidities, no vital sign noise, no patient non-compliance.
- Agent has 5 steps to diagnose → good for testing basic reasoning.
"""


class EasyTask:
    task_id = "easy_diagnosis"
    name = "Easy Diagnosis"
    diseases = ["severe_anaemia", "neonatal_jaundice", "low_birth_weight", "hyperemesis", "hypothermia_newborn"]
    max_steps = 5
    difficulty = "easy"
    all_symptoms_revealed = True
    comorbidity = False
    noise = False
