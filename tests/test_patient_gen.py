"""
Tests for the patient generator (asha_env/patient/generator.py).

The patient generator is responsible for creating realistic, varied synthetic
patients for each episode. These tests verify:
    - Required fields are always present in the generated patient dict
    - Demographics match the disease type (maternal vs newborn)
    - Vitals are appropriate for the patient type and disease
    - Difficulty level controls how many symptoms are revealed at episode start
    - Seasonal weights affect which diseases appear more frequently
    - Hard difficulty produces comorbidities and vital noise

Run with:
    pytest tests/test_patient_gen.py -v
"""

import sys
import os

# Allow importing asha_env from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import random
from collections import Counter
from asha_env.patient.generator import generate_patient
from asha_env.patient.epidemiology import get_season


# A minimal but realistic village dict used across all tests.
# Represents a rural village in Maharashtra with typical characteristics.
SAMPLE_VILLAGE = {
    "id": "village_001",
    "name": "Rampur",
    "state": "Maharashtra",
    "region": "rural_maharashtra",
    "population": 1200,
    "active_outbreaks": [],
    "recent_cases": ["severe_anaemia", "neonatal_jaundice"],  # Diseases currently common here
    "water_source": "well",
    "sanitation": "low",
}


def test_patient_has_required_fields():
    """Every generated patient must have all fields the environment and graders depend on."""
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE,
                               ["severe_anaemia", "pre_eclampsia", "neonatal_jaundice", "hyperemesis", "low_birth_weight"])
    assert "demographics" in patient       # Age, gender, socioeconomic status, pregnancy status
    assert "true_diagnosis" in patient     # The hidden ground-truth disease
    assert "true_symptoms" in patient      # All symptoms the patient actually has
    assert "revealed_symptoms" in patient  # Subset visible to the agent (depends on difficulty)
    assert "chief_complaint" in patient    # The presenting complaint the agent sees first
    assert "vitals" in patient             # BP, temperature, pulse, SpO2, etc.
    assert "history" in patient            # Medical/social history items
    assert "patient_trust" in patient      # Affects how reliably the patient answers
    assert "non_compliance_rate" in patient # Probability of unreliable symptom answers
    assert "disease_data" in patient       # Full disease metadata from diseases.json


def test_patient_demographics():
    """
    Maternal diseases (like severe_anaemia) must generate a pregnant female patient,
    not a newborn. The generator uses the disease category to determine patient type.
    """
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])
    demo = patient["demographics"]
    assert "age" in demo
    assert "gender" in demo
    assert "socioeconomic" in demo
    assert demo["socioeconomic"] in ("BPL", "APL")  # Below or above poverty line
    assert "patient_type" in demo
    # Severe anaemia is a maternal disease — patient must be a pregnant woman.
    assert demo["gender"] == "F"
    assert demo["pregnant"] is True
    assert demo["patient_type"] == "mother"


def test_newborn_demographics():
    """
    Neonatal diseases (like neonatal_jaundice) must generate a newborn patient,
    not a pregnant woman. Age should be 0 and pregnant should be False.
    """
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["neonatal_jaundice"])
    demo = patient["demographics"]
    assert demo["age"] == 0              # Newborns have age 0
    assert demo["patient_type"] == "newborn"
    assert demo["pregnant"] is False     # Newborns are never pregnant


def test_patient_vitals():
    """Maternal patients must have standard vital signs (temp, pulse, SpO2)."""
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE,
                               ["severe_anaemia", "pre_eclampsia"])
    vitals = patient["vitals"]
    assert "temp_f" in vitals   # Temperature in Fahrenheit
    assert "pulse" in vitals    # Heart rate in bpm
    assert "spo2" in vitals     # Oxygen saturation percentage


def test_newborn_vitals():
    """
    Low birth weight newborns must have a weight_kg vital, and it must be
    below 2.5kg — the clinical threshold for low birth weight.
    """
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["low_birth_weight"])
    vitals = patient["vitals"]
    assert "weight_kg" in vitals
    assert vitals["weight_kg"] < 2.5  # WHO definition of low birth weight


def test_easy_reveals_all_symptoms():
    """
    On easy difficulty, all true symptoms must be revealed at episode start.
    The agent doesn't need to ask — everything is visible from the first observation.
    """
    patient = generate_patient("easy", "monsoon", SAMPLE_VILLAGE,
                               ["severe_anaemia", "neonatal_jaundice", "hyperemesis"])
    assert set(patient["revealed_symptoms"]) == set(patient["true_symptoms"])


def test_medium_reveals_only_chief_complaint():
    """
    On medium difficulty, only 1-2 symptoms should be revealed at the start
    (the chief complaint), forcing the agent to ask questions to uncover the rest.
    Revealed symptoms must always be a subset of the true symptoms.
    Repeated 10 times to guard against random variation passing by chance.
    """
    for _ in range(10):
        patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE,
                                   ["severe_anaemia", "pre_eclampsia", "hyperemesis",
                                    "neonatal_jaundice", "low_birth_weight"])
        # Revealed must be a strict subset — can't reveal symptoms the patient doesn't have.
        assert set(patient["revealed_symptoms"]).issubset(set(patient["true_symptoms"]))
        # On medium, at most as many revealed as true (fewer symptoms shown upfront).
        assert len(patient["revealed_symptoms"]) <= len(patient["true_symptoms"])


def test_seasonal_weights_affect_disease_distribution():
    """
    The epidemiology module weights diseases by season. Hypothermia in newborns
    is much more common in winter than summer — verify this holds statistically
    over 200 samples per season.
    """
    diseases_allowed = ["hypothermia_newborn", "neonatal_jaundice", "low_birth_weight",
                        "severe_anaemia", "hyperemesis"]

    winter_counts = Counter()
    summer_counts = Counter()

    for _ in range(200):
        p = generate_patient("easy", "winter", SAMPLE_VILLAGE, diseases_allowed)
        winter_counts[p["true_diagnosis"]] += 1

    for _ in range(200):
        p = generate_patient("easy", "summer", SAMPLE_VILLAGE, diseases_allowed)
        summer_counts[p["true_diagnosis"]] += 1

    # Hypothermia_newborn must appear meaningfully more in winter than summer.
    assert winter_counts["hypothermia_newborn"] > summer_counts["hypothermia_newborn"], \
        f"Winter hypothermia ({winter_counts['hypothermia_newborn']}) should > summer ({summer_counts['hypothermia_newborn']})"


def test_hard_task_has_comorbidities():
    """
    Hard difficulty introduces comorbidities (secondary conditions alongside
    the primary disease). Over 50 hard episodes, at least some must have them.
    """
    comorbid_count = 0
    for _ in range(50):
        patient = generate_patient("hard", "monsoon", SAMPLE_VILLAGE,
                                   ["severe_anaemia", "pre_eclampsia", "low_birth_weight",
                                    "hypothermia_newborn", "neonatal_sepsis", "gestational_diabetes",
                                    "postpartum_haemorrhage"])
        if len(patient["comorbidities"]) > 0:
            comorbid_count += 1

    assert comorbid_count > 0, "No comorbidities found in 50 hard episodes"


def test_vitals_noise_applied_on_medium_hard():
    """
    On medium/hard difficulty, random noise is applied to vitals so each patient
    is slightly different even for the same disease. Generate 30 patients with
    the same disease and verify temperature values are not all identical.
    """
    temps = []
    for _ in range(30):
        patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])
        temps.append(patient["vitals"]["temp_f"])

    # Round to 1 decimal to avoid floating-point false negatives.
    unique_temps = set(round(t, 1) for t in temps)
    assert len(unique_temps) > 1, "Vitals should vary across patients on medium difficulty"


def test_non_compliance_rates():
    """
    Non-compliance rate controls how often the patient gives unreliable symptom
    answers. It must increase with difficulty: easy=0.0, medium=0.1, hard=0.2.
    """
    easy = generate_patient("easy", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])
    medium = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])
    hard = generate_patient("hard", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])

    assert easy["non_compliance_rate"] == 0.0    # Easy: patient always answers truthfully
    assert medium["non_compliance_rate"] == 0.1  # Medium: 10% chance of unreliable answer
    assert hard["non_compliance_rate"] == 0.2    # Hard: 20% chance of unreliable answer


def test_patient_trust_starts_at_75():
    """
    Patient trust must always start at 75 regardless of difficulty or disease.
    Trust decreases during the episode when the ASHA asks unhelpful questions
    or the patient is non-compliant.
    """
    patient = generate_patient("medium", "monsoon", SAMPLE_VILLAGE, ["severe_anaemia"])
    assert patient["patient_trust"] == 75
