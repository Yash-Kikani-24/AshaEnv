class HardTask:
    task_id = "hard_complex_case"
    name = "Hard Complex Case"
    diseases = [
        "malaria", "dengue", "typhoid", "tuberculosis", "anaemia",
        "pneumonia", "diarrhoea", "malnutrition", "hypertension", "diabetes",
        "chickenpox", "jaundice", "worm_infestation", "pre_eclampsia", "ari",
    ]
    max_steps = 20
    difficulty = "hard"
    all_symptoms_revealed = False
    comorbidity_chance = 0.30
    non_compliance_rate = 0.20
    emergency_cases = True
    vitals_noise = 0.05
