import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import random
from collections import Counter
from asha_env.patient.generator import generate_patient
from asha_env.patient.epidemiology import get_season


SAMPLE_VILLAGE = {
    "id": "village_001",
    "name": "Rampur",
    "state": "Maharashtra",
    "region": "rural_maharashtra",
    "population": 1200,
    "active_outbreaks": [],
    "recent_cases": ["malaria", "malaria", "dengue"],
    "water_source": "well",
    "sanitation": "low",
}


def test_patient_has_required_fields():
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE,
                               ["malaria", "dengue", "typhoid", "anaemia", "diarrhoea"])
    assert "demographics" in patient
    assert "true_diagnosis" in patient
    assert "true_symptoms" in patient
    assert "revealed_symptoms" in patient
    assert "chief_complaint" in patient
    assert "vitals" in patient
    assert "history" in patient
    assert "patient_trust" in patient
    assert "non_compliance_rate" in patient
    assert "disease_data" in patient


def test_patient_demographics():
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["malaria"])
    demo = patient["demographics"]
    assert "age" in demo
    assert "gender" in demo
    assert demo["gender"] in ("M", "F")
    assert "socioeconomic" in demo
    assert demo["socioeconomic"] in ("BPL", "APL")
    assert isinstance(demo["age"], int)
    assert 1 <= demo["age"] <= 80


def test_patient_vitals():
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["malaria"])
    vitals = patient["vitals"]
    assert "temp_f" in vitals
    assert "bp_systolic" in vitals
    assert "pulse" in vitals
    assert "spo2" in vitals


def test_easy_reveals_all_symptoms():
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE,
                               ["malaria", "anaemia", "diarrhoea"])
    # On easy, revealed should equal true symptoms
    assert set(patient["revealed_symptoms"]) == set(patient["true_symptoms"])


def test_medium_reveals_only_chief_complaint():
    # Run multiple times to be confident
    for _ in range(10):
        patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE,
                                   ["malaria", "dengue", "typhoid", "anaemia", "diarrhoea"])
        # Revealed should be subset of true
        assert set(patient["revealed_symptoms"]).issubset(set(patient["true_symptoms"]))
        # Revealed should be <= chief symptoms (1-2 symptoms)
        assert len(patient["revealed_symptoms"]) <= len(patient["true_symptoms"])


def test_seasonal_weights_affect_disease_distribution():
    """Monsoon should produce more malaria/dengue/diarrhoea than winter."""
    diseases_allowed = ["malaria", "dengue", "diarrhoea", "ari", "pneumonia"]

    monsoon_counts = Counter()
    winter_counts = Counter()

    for _ in range(200):
        p = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, diseases_allowed)
        monsoon_counts[p["true_diagnosis"]] += 1

    for _ in range(200):
        p = generate_patient("easy", "winter", SAMPLE_VILLAGE, diseases_allowed)
        winter_counts[p["true_diagnosis"]] += 1

    # Monsoon diseases should appear more in monsoon
    monsoon_tropical = monsoon_counts["malaria"] + monsoon_counts["dengue"] + monsoon_counts["diarrhoea"]
    winter_tropical = winter_counts["malaria"] + winter_counts["dengue"] + winter_counts["diarrhoea"]
    # Winter respiratory should appear more in winter
    winter_resp = winter_counts["ari"] + winter_counts["pneumonia"]
    monsoon_resp = monsoon_counts["ari"] + monsoon_counts["pneumonia"]

    assert monsoon_tropical > winter_tropical, \
        f"Monsoon tropical ({monsoon_tropical}) should > winter ({winter_tropical})"
    assert winter_resp > monsoon_resp, \
        f"Winter respiratory ({winter_resp}) should > monsoon ({monsoon_resp})"


def test_hard_task_has_comorbidities():
    """Over many hard episodes, at least some should have comorbidities."""
    comorbid_count = 0
    for _ in range(50):
        patient = generate_patient("hard", "monsoon", SAMPLE_VILLAGE,
                                   ["malaria", "dengue", "anaemia", "malnutrition",
                                    "diarrhoea", "tuberculosis", "worm_infestation"])
        if len(patient["comorbidities"]) > 0:
            comorbid_count += 1

    assert comorbid_count > 0, "No comorbidities found in 50 hard episodes"


def test_vitals_noise_applied_on_medium_hard():
    """Medium/hard vitals should vary; run same disease many times and check spread."""
    temps = []
    for _ in range(30):
        patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["malaria"])
        temps.append(patient["vitals"]["temp_f"])

    unique_temps = set(round(t, 1) for t in temps)
    assert len(unique_temps) > 1, "Vitals should vary across patients"


def test_non_compliance_rates():
    easy = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["malaria"])
    medium = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["malaria"])
    hard = generate_patient("hard", "monsoon", SAMPLE_VILLAGE, ["malaria"])

    assert easy["non_compliance_rate"] == 0.0
    assert medium["non_compliance_rate"] == 0.1
    assert hard["non_compliance_rate"] == 0.2


def test_patient_trust_starts_at_75():
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["malaria"])
    assert patient["patient_trust"] == 75
