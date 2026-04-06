"""
Patient generator — creates a fully specified synthetic patient for each episode.

Generates: demographics, disease (weighted by season/region/outbreaks), vitals,
symptoms (with partial reveal based on difficulty), medical history, and
behavioral parameters like non-compliance rate and trust level.

The generated patient dict is the "ground truth" for the episode. The environment
selectively reveals parts of it to the agent via observations.
"""

import json
import os
import random

from .epidemiology import get_disease_priors, load_diseases
from .comorbidity import get_comorbidities

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Diseases that affect newborns (age=0) rather than the mother.
# Used to decide demographics (newborn vs. pregnant woman) and vitals ranges.
NEWBORN_DISEASES = {
    "birth_asphyxia", "neonatal_sepsis", "neonatal_jaundice",
    "low_birth_weight", "hypothermia_newborn",
}


def _load_symptoms():
    """Load symptoms.json → dict keyed by symptom_id."""
    with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
        return json.load(f)


def _load_kit():
    """Load asha_kit.json → dict with 'medicines', 'tests', 'supplies' lists."""
    with open(os.path.join(DATA_DIR, "asha_kit.json")) as f:
        return json.load(f)


def _pick_disease(allowed_diseases: list[str], season: str, village: dict) -> str:
    """
    Select a primary disease using weighted random sampling.
    Weights come from epidemiological priors (season + region + outbreaks).
    Falls back to uniform random if no priors match the allowed list.
    """
    priors = get_disease_priors(season, village["region"], village.get("active_outbreaks", []))
    # Filter to allowed diseases
    filtered = {k: v for k, v in priors.items() if k in allowed_diseases}
    if not filtered:
        return random.choice(allowed_diseases)
    diseases = list(filtered.keys())
    weights = list(filtered.values())
    return random.choices(diseases, weights=weights, k=1)[0]


def _generate_vitals(disease: dict, difficulty: str) -> dict:
    """
    Generate realistic vital signs for the patient based on their disease.

    Newborns get: temp, pulse, resp_rate, spo2, weight_kg.
    Mothers get:  temp, bp_systolic, bp_diastolic, pulse, spo2, resp_rate.

    Each disease modifies specific vitals to abnormal ranges (e.g., eclampsia → high BP).
    On medium/hard difficulty, ±5% random noise is added to all vitals to simulate
    measurement imprecision from basic ASHA kit instruments.
    """
    disease_id = disease["id"]
    is_newborn = disease_id in NEWBORN_DISEASES

    if is_newborn:
        base_vitals = {
            "temp_f": 98.6,
            "pulse": 130,
            "resp_rate": 40,
            "spo2": 97,
            "weight_kg": 3.0,
        }
        if disease_id == "birth_asphyxia":
            base_vitals["spo2"] = random.randint(70, 85)
            base_vitals["pulse"] = random.randint(60, 90)
            base_vitals["resp_rate"] = random.randint(5, 15)
        elif disease_id == "neonatal_sepsis":
            base_vitals["temp_f"] = random.uniform(100.0, 103.0)
            base_vitals["pulse"] = random.randint(160, 190)
            base_vitals["resp_rate"] = random.randint(55, 75)
        elif disease_id == "hypothermia_newborn":
            base_vitals["temp_f"] = random.uniform(93.0, 96.5)
            base_vitals["pulse"] = random.randint(80, 110)
        elif disease_id == "low_birth_weight":
            base_vitals["weight_kg"] = round(random.uniform(1.5, 2.4), 1)
            base_vitals["temp_f"] = random.uniform(96.0, 98.0)
        elif disease_id == "neonatal_jaundice":
            base_vitals["temp_f"] = 98.6
    else:
        base_vitals = {
            "temp_f": 98.6,
            "bp_systolic": 120,
            "bp_diastolic": 80,
            "pulse": 80,
            "spo2": 98,
            "resp_rate": 18,
        }
        if disease_id in ("pre_eclampsia", "eclampsia"):
            base_vitals["bp_systolic"] = random.randint(145, 180)
            base_vitals["bp_diastolic"] = random.randint(90, 115)
            base_vitals["pulse"] = random.randint(85, 105)
        elif disease_id in ("antepartum_haemorrhage", "postpartum_haemorrhage"):
            base_vitals["pulse"] = random.randint(100, 130)
            base_vitals["bp_systolic"] = random.randint(80, 100)
            base_vitals["bp_diastolic"] = random.randint(50, 65)
            base_vitals["spo2"] = random.randint(92, 96)
        elif disease_id == "puerperal_sepsis":
            base_vitals["temp_f"] = random.uniform(100.5, 103.5)
            base_vitals["pulse"] = random.randint(95, 120)
        elif disease_id == "severe_anaemia":
            base_vitals["pulse"] = random.randint(95, 120)
            base_vitals["spo2"] = random.randint(93, 97)
        elif disease_id == "gestational_diabetes":
            base_vitals["pulse"] = random.randint(80, 95)
        elif disease_id == "obstructed_labour":
            base_vitals["pulse"] = random.randint(100, 125)
            base_vitals["temp_f"] = random.uniform(99.0, 101.0)
        elif disease_id == "preterm_labour":
            base_vitals["pulse"] = random.randint(85, 105)

    # Add noise on medium/hard
    if difficulty in ("medium", "hard"):
        noise_pct = 0.05
        for key in base_vitals:
            noise = base_vitals[key] * random.uniform(-noise_pct, noise_pct)
            base_vitals[key] = round(base_vitals[key] + noise, 1)

    return base_vitals


def _build_symptom_lists(disease: dict, comorbid_diseases: list[dict], difficulty: str):
    """
    Build the patient's symptom profile.

    Returns (true_symptoms, revealed_symptoms, chief_symptoms):
      - true_symptoms: all symptoms the patient actually has (required + random optional + comorbid)
      - revealed_symptoms: what the agent sees upfront (all on easy, only chief on medium/hard)
      - chief_symptoms: the 1-2 most prominent symptoms used for the chief complaint text
    """
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


def _generate_demographics(disease_id: str) -> dict:
    """
    Generate patient demographics based on disease type.
    - Newborn diseases → age=0, random gender, patient_type="newborn"
    - Maternal diseases → age 18-40, female, pregnant, patient_type="mother"
    Socioeconomic status is 60% BPL (Below Poverty Line), 40% APL (Above Poverty Line).
    """
    is_newborn = disease_id in NEWBORN_DISEASES

    if is_newborn:
        gender = random.choice(["M", "F"])
        age = 0
        socioeconomic = random.choices(["BPL", "APL"], weights=[60, 40], k=1)[0]
        return {
            "age": age,
            "gender": gender,
            "socioeconomic": socioeconomic,
            "pregnant": False,
            "patient_type": "newborn",
        }
    else:
        age = random.randint(18, 40)
        socioeconomic = random.choices(["BPL", "APL"], weights=[60, 40], k=1)[0]
        return {
            "age": age,
            "gender": "F",
            "socioeconomic": socioeconomic,
            "pregnant": True,
            "patient_type": "mother",
        }


def _generate_history(disease: dict) -> list[str]:
    """
    Generate a random subset of medical/social history items relevant to the disease.
    Each item in the disease's history pool has a 60% chance of being included.
    These are things the agent can discover by using "ask_history:<item>" actions.
    """
    history_items = []
    disease_id = disease["id"]

    # Maps each disease to its pool of plausible history risk factors
    history_pool = {
        "severe_anaemia": ["poor_diet", "no_ifa_supplements", "multiple_pregnancies", "vegetarian_diet"],
        "pre_eclampsia": ["first_pregnancy", "family_history_bp", "previous_pre_eclampsia", "teen_pregnancy", "age_over_35"],
        "eclampsia": ["no_anc_visits", "previous_pre_eclampsia", "first_pregnancy", "family_history_bp"],
        "antepartum_haemorrhage": ["previous_caesarean", "placenta_previa_history", "multiple_pregnancies", "smoking_in_household"],
        "postpartum_haemorrhage": ["grand_multipara", "prolonged_labour_history", "previous_pph", "anaemia_in_pregnancy"],
        "puerperal_sepsis": ["home_delivery", "prolonged_labour_history", "premature_rupture_membranes", "no_clean_delivery_kit"],
        "gestational_diabetes": ["family_history_diabetes", "previous_gdm", "obesity", "age_over_35", "previous_large_baby"],
        "hyperemesis": ["first_pregnancy", "multiple_gestation", "previous_hyperemesis"],
        "preterm_labour": ["previous_preterm", "multiple_gestation", "cervical_incompetence", "uti_in_pregnancy"],
        "obstructed_labour": ["short_stature", "teen_pregnancy", "first_pregnancy", "no_anc_visits"],
        "birth_asphyxia": ["prolonged_labour_history", "home_delivery", "no_skilled_attendant", "premature_birth"],
        "neonatal_sepsis": ["home_delivery", "premature_rupture_membranes", "maternal_fever_during_labour", "unclean_cord_care"],
        "neonatal_jaundice": ["premature_birth", "blood_group_incompatibility", "breastfeeding_difficulty"],
        "low_birth_weight": ["poor_diet", "anaemia_in_pregnancy", "teen_pregnancy", "no_anc_visits", "multiple_gestation"],
        "hypothermia_newborn": ["premature_birth", "low_birth_weight_history", "home_delivery", "cold_environment"],
    }

    pool = history_pool.get(disease_id, [])
    for item in pool:
        if random.random() < 0.6:
            history_items.append(item)

    return history_items


def generate_patient(task_difficulty: str, season: str, village: dict, allowed_diseases: list[str]) -> dict:
    """
    Main entry point — generate a complete synthetic patient for one episode.

    Returns a dict containing all ground-truth data:
      demographics, true_diagnosis, comorbidities, true_symptoms, revealed_symptoms,
      chief_complaint, vitals, history, trust level, non_compliance_rate, etc.

    The environment's _build_observation() method controls what subset the agent actually sees.
    """
    diseases_db = load_diseases()

    # Pick primary disease weighted by season/region/outbreaks
    primary_id = _pick_disease(allowed_diseases, season, village)
    primary_disease = diseases_db[primary_id]

    # Get comorbidities
    comorbid_ids = get_comorbidities(primary_id, task_difficulty)
    comorbid_diseases = [diseases_db[cid] for cid in comorbid_ids if cid in diseases_db]

    # Demographics based on disease type
    demographics = _generate_demographics(primary_id)

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
    if demographics["patient_type"] == "newborn":
        chief_complaint = f"newborn with {', '.join(chief_names).lower()}"
    else:
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
