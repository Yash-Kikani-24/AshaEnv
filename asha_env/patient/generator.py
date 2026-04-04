import json
import os
import random

from .epidemiology import get_disease_priors, load_diseases
from .comorbidity import get_comorbidities

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_symptoms():
    with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
        return json.load(f)


def _load_kit():
    with open(os.path.join(DATA_DIR, "asha_kit.json")) as f:
        return json.load(f)


def _pick_disease(allowed_diseases: list[str], season: str, village: dict) -> str:
    priors = get_disease_priors(season, village["region"], village.get("active_outbreaks", []))
    # Filter to allowed diseases
    filtered = {k: v for k, v in priors.items() if k in allowed_diseases}
    if not filtered:
        return random.choice(allowed_diseases)
    diseases = list(filtered.keys())
    weights = list(filtered.values())
    return random.choices(diseases, weights=weights, k=1)[0]


def _generate_vitals(disease: dict, difficulty: str) -> dict:
    base_vitals = {
        "temp_f": 98.6,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "pulse": 72,
        "spo2": 98,
        "resp_rate": 16,
    }

    # Adjust vitals based on disease
    disease_id = disease["id"]
    if disease_id in ("malaria", "dengue", "typhoid", "pneumonia", "ari", "chickenpox"):
        base_vitals["temp_f"] = random.uniform(100.5, 104.0)
        base_vitals["pulse"] = random.randint(88, 110)
    if disease_id in ("pneumonia", "ari"):
        base_vitals["spo2"] = random.randint(90, 96)
        base_vitals["resp_rate"] = random.randint(22, 32)
    if disease_id in ("hypertension", "pre_eclampsia"):
        base_vitals["bp_systolic"] = random.randint(145, 180)
        base_vitals["bp_diastolic"] = random.randint(90, 110)
    if disease_id == "anaemia":
        base_vitals["pulse"] = random.randint(90, 115)
        base_vitals["spo2"] = random.randint(94, 97)
    if disease_id == "diarrhoea":
        base_vitals["pulse"] = random.randint(85, 105)

    # Add noise on medium/hard
    if difficulty in ("medium", "hard"):
        noise_pct = 0.05
        for key in base_vitals:
            noise = base_vitals[key] * random.uniform(-noise_pct, noise_pct)
            base_vitals[key] = round(base_vitals[key] + noise, 1)

    return base_vitals


def _build_symptom_lists(disease: dict, comorbid_diseases: list[dict], difficulty: str):
    all_symptoms = load_diseases()
    true_symptoms = list(disease["required_symptoms"])

    # Add some optional symptoms
    for s in disease["optional_symptoms"]:
        if random.random() < 0.5:
            true_symptoms.append(s)

    # Add comorbid symptoms
    for cd in comorbid_diseases:
        for s in cd["required_symptoms"]:
            if s not in true_symptoms:
                true_symptoms.append(s)
        for s in cd["optional_symptoms"]:
            if s not in true_symptoms and random.random() < 0.3:
                true_symptoms.append(s)

    # Chief complaint: 1-2 most prominent symptoms
    chief_symptoms = list(disease["required_symptoms"][:2])

    if difficulty == "easy":
        # All symptoms revealed
        revealed = list(true_symptoms)
    else:
        # Only chief complaint revealed
        revealed = list(chief_symptoms)

    return true_symptoms, revealed, chief_symptoms


def _generate_demographics() -> dict:
    gender = random.choice(["M", "F"])
    age = random.choices(
        [random.randint(1, 5), random.randint(6, 14), random.randint(15, 45),
         random.randint(46, 65), random.randint(66, 80)],
        weights=[15, 15, 40, 20, 10],
        k=1,
    )[0]
    socioeconomic = random.choices(["BPL", "APL"], weights=[60, 40], k=1)[0]
    pregnant = gender == "F" and 18 <= age <= 40 and random.random() < 0.15
    return {
        "age": age,
        "gender": gender,
        "socioeconomic": socioeconomic,
        "pregnant": pregnant,
    }


def _generate_history(disease: dict) -> list[str]:
    history_items = []
    disease_id = disease["id"]

    history_pool = {
        "malaria": ["recent_travel", "mosquito_exposure", "previous_malaria"],
        "dengue": ["mosquito_exposure", "neighbourhood_cases"],
        "typhoid": ["contaminated_water", "street_food"],
        "tuberculosis": ["tb_contact", "previous_tb", "overcrowded_living"],
        "anaemia": ["poor_diet", "heavy_menstruation", "vegetarian_diet"],
        "pneumonia": ["smoking", "recent_cold", "overcrowded_living"],
        "diarrhoea": ["contaminated_water", "street_food", "no_handwashing"],
        "malnutrition": ["poor_diet", "poverty", "food_insecurity"],
        "hypertension": ["family_history_bp", "salt_heavy_diet", "stress"],
        "diabetes": ["family_history_diabetes", "sedentary_lifestyle", "obesity"],
        "chickenpox": ["school_exposure", "no_vaccination"],
        "jaundice": ["contaminated_water", "alcohol_use", "previous_hepatitis"],
        "worm_infestation": ["barefoot_walking", "no_deworming", "poor_hygiene"],
        "pre_eclampsia": ["first_pregnancy", "family_history_bp", "previous_pre_eclampsia"],
        "ari": ["cold_exposure", "overcrowded_living", "smoking_in_household"],
    }

    pool = history_pool.get(disease_id, [])
    for item in pool:
        if random.random() < 0.6:
            history_items.append(item)

    return history_items


def generate_patient(task_difficulty: str, season: str, village: dict, allowed_diseases: list[str]) -> dict:
    diseases_db = load_diseases()

    # Pick primary disease
    primary_id = _pick_disease(allowed_diseases, season, village)
    primary_disease = diseases_db[primary_id]

    # Get comorbidities
    comorbid_ids = get_comorbidities(primary_id, task_difficulty)
    comorbid_diseases = [diseases_db[cid] for cid in comorbid_ids if cid in diseases_db]

    # Demographics
    demographics = _generate_demographics()

    # Force pregnancy for pre_eclampsia
    if primary_id == "pre_eclampsia":
        demographics["gender"] = "F"
        demographics["age"] = random.randint(18, 38)
        demographics["pregnant"] = True

    # Symptoms
    true_symptoms, revealed_symptoms, chief_symptoms = _build_symptom_lists(
        primary_disease, comorbid_diseases, task_difficulty
    )

    # Vitals
    vitals = _generate_vitals(primary_disease, task_difficulty)

    # History
    history = _generate_history(primary_disease)

    # Chief complaint text
    symptom_db = _load_symptoms()
    chief_names = [symptom_db[s]["name"] for s in chief_symptoms if s in symptom_db]
    chief_complaint = f"{', '.join(chief_names).lower()} for {random.randint(1, 7)} days"

    # Non-compliance rate
    non_compliance_rate = {"easy": 0.0, "medium": 0.1, "hard": 0.2}.get(task_difficulty, 0.0)

    return {
        "demographics": demographics,
        "true_diagnosis": primary_id,
        "comorbidities": comorbid_ids,
        "true_symptoms": true_symptoms,
        "revealed_symptoms": list(revealed_symptoms),
        "chief_complaint": chief_complaint,
        "chief_symptoms": chief_symptoms,
        "vitals": vitals,
        "history": history,
        "revealed_history": [],
        "patient_trust": 75,
        "non_compliance_rate": non_compliance_rate,
        "village": village,
        "season": season,
        "disease_data": primary_disease,
    }
