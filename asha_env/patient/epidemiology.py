import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_diseases():
    with open(os.path.join(DATA_DIR, "diseases.json")) as f:
        return {d["id"]: d for d in json.load(f)}


def get_season(month: int = None) -> str:
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
    diseases = load_diseases()
    priors = {}

    for did, d in diseases.items():
        weight = d["seasonal_weights"].get(season, 1.0)

        # Regional boosts
        if region == "rural_bihar" and did in ("malaria", "anaemia", "malnutrition"):
            weight *= 1.5
        elif region == "rural_up" and did in ("tuberculosis", "ari", "pneumonia"):
            weight *= 1.3
        elif region == "rural_maharashtra" and did in ("dengue", "typhoid"):
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
