class MediumTask:
    task_id = "medium_consultation"
    name = "Medium Consultation"
    diseases = [
        "malaria", "dengue", "typhoid", "anaemia", "pneumonia",
        "diarrhoea", "malnutrition", "chickenpox", "worm_infestation", "ari",
    ]
    max_steps = 12
    difficulty = "medium"
    all_symptoms_revealed = False
    comorbidity_chance = 0.1
    vitals_noise = 0.05
