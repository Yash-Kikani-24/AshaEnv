"""
Epidemiology helpers — season detection and disease probability priors.

Determines which diseases are more likely based on:
  - Current season (monsoon, winter, summer)
  - Geographic region (rural Bihar, UP, Maharashtra each have different risk profiles)
  - Active outbreaks in the village (3x boost)

Used by the patient generator to pick a realistic primary disease.
"""

import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_diseases():
    """Load diseases.json and return a dict keyed by disease ID."""
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


def get_season(month: int = None) -> str:
    """Map a month (1-12) to an Indian season. Defaults to the current month."""
    if month is None:
        month = datetime.now().month
    if month in (6, 7, 8, 9):
        return "monsoon"
    elif month in (11, 12, 1, 2):
        return "winter"
    else:
        return "summer"


def get_disease_priors(
    season: str,
    region: str,
    active_outbreaks: list[str],
) -> dict[str, float]:
    """
    Compute a probability distribution over all diseases given context.

    Weights are built by multiplying:
      1. Seasonal weight from diseases.json (e.g., anaemia is equally common year-round)
      2. Regional multiplier (e.g., Bihar has higher anaemia prevalence)
      3. Outbreak multiplier (3x if the disease has an active outbreak in the village)

    Returns a normalized dict {disease_id: probability}.
    """
    diseases = load_diseases()
    priors = {}

    for did, d in diseases.items():
        weight = d["seasonal_weights"].get(season, 1.0)

        # Regional boosts — reflect real epidemiological differences across Indian states
        if region == "rural_bihar" and did in ("severe_anaemia", "low_birth_weight", "hypothermia_newborn"):
            weight *= 1.5
        elif region == "rural_up" and did in ("eclampsia", "obstructed_labour", "pre_eclampsia"):
            weight *= 1.3
        elif region == "rural_maharashtra" and did in ("gestational_diabetes", "neonatal_jaundice"):
            weight *= 1.3

        # Outbreak boost
        if did in active_outbreaks:
            weight *= 3.0

        priors[did] = weight

    # Normalize to probabilities
    total = sum(priors.values())
    if total > 0:
        priors = {k: v / total for k, v in priors.items()}

    return priors
