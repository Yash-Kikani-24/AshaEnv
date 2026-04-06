---
title: ASHA Agent Environment
emoji: "\U0001F3E5"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# ASHA Agent Environment

An RL training environment that simulates patient consultations for **ASHA (Accredited Social Health Activist) workers** in rural India. Built for the **Meta PyTorch x Scaler School of Technology OpenEnv Hackathon**.

## What problem does this solve?

Over 600 million people in rural India depend on a network of ~1 million ASHA workers as their first point of healthcare contact. These community health workers operate with limited medical kits and must make critical decisions: Which symptoms to investigate? When to treat locally vs. refer to a Primary Health Centre? How to handle emergencies like pre-eclampsia with no doctor nearby?

This environment lets AI agents practice these exact decisions in a realistic simulation with seasonal disease patterns, regional epidemiology, comorbidities, patient non-compliance, and resource constraints, creating a safe space to learn clinical decision-making before any real-world deployment.

## Environment Description

The agent interacts with a simulated patient through a turn-based consultation:

| Action Type | Format | Description |
|-------------|--------|-------------|
| Ask Symptom | `ask_symptom:fever` | Ask if the patient has a specific symptom |
| Ask History | `ask_history:recent_travel` | Ask about patient history |
| Order Test | `order_test:malaria_rdt` | Run a test from the ASHA kit |
| Diagnose | `diagnose:malaria` | Make a diagnosis (ends episode) |
| Treat | `treat:chloroquine` | Administer medicine from kit |
| Refer | `refer:phc` | Refer to facility (ends episode) |

### Observation Fields

| Field | Description |
|-------|-------------|
| `patient.age`, `gender`, `chief_complaint` | Basic demographics |
| `patient.known_symptoms` | Symptoms revealed so far |
| `patient.vitals` | Temperature, BP, SpO2, pulse |
| `patient.village_context` | Season, outbreaks, water source |
| `asha_context.kit_available` | Medicines in the ASHA kit |
| `asha_context.patient_trust` | Trust level (affects compliance) |
| `available_actions` | Valid actions for current state |

## Tasks

| Task | Diseases | Difficulty | Max Steps | Key Challenge |
|------|----------|------------|-----------|---------------|
| `easy_diagnosis` | 5 | Easy | 5 | All symptoms visible, just diagnose |
| `medium_consultation` | 10 | Medium | 12 | Partial observability, must ask questions |
| `hard_complex_case` | 15 | Hard | 20 | Comorbidities, emergencies, non-compliance |

## Quickstart

```bash
# Build and run locally
docker build -t asha-env .
docker run -p 7860:7860 asha-env

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium_consultation"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ask_symptom:fever", "episode_id": "EPISODE_ID_FROM_RESET"}'

# Check full state (including hidden diagnosis)
curl http://localhost:7860/state
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode (optional `task_id`) |
| POST | `/step` | Take action (`action` + `episode_id`) |
| GET | `/state` | Full state including true diagnosis |
| GET | `/health` | Health check |

## Reward Design

The environment provides step-level rewards to guide learning:

- **Correct diagnosis**: +1.0 (with speed bonus for fewer steps)
- **Same-category diagnosis**: +0.3
- **Wrong diagnosis**: -0.3 (or -1.0 if emergency missed)
- **Relevant symptom found**: +0.05 * specificity
- **Relevant test ordered**: +0.10
- **Correct referral**: +0.8
- **Over-referral**: +0.3 (safe but wasteful)
- **Under-referral on emergency**: -1.0
- **Invalid action**: -0.1

## Graders

The composite grader scores complete episodes (0.0 to 1.0):

| Grader | Weight | What it measures |
|--------|--------|------------------|
| **Diagnosis** | 40% | Correct diagnosis, category match, emergency detection |
| **Safety** | 25% | Emergency referral, harmful treatment avoidance |
| **Efficiency** | 20% | Questions asked vs. minimum needed, relevance |
| **Referral** | 15% | Correct referral level, over/under-referral |

## Baseline Scores

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | ~0.50 | ~0.49 | ~0.45 | ~0.48 |
| Rule-based | ~0.83 | ~0.48 | ~0.62 | ~0.64 |

## Diseases Covered

Malaria, Dengue, Typhoid, Tuberculosis, Anaemia, Pneumonia, Diarrhoea/Cholera, Malnutrition, Hypertension, Diabetes, Chickenpox, Jaundice/Hepatitis, Worm Infestation, Pre-eclampsia, Acute Respiratory Infection

## ASHA Kit

**Medicines**: ORS, Iron tablets, OCP, Chloroquine, Paracetamol, Zinc, Albendazole
**Tests**: Malaria RDT, Haemoglobin strip, Urine dipstick, BP monitor, Thermometer

## HuggingFace Space

Live at: [https://huggingface.co/spaces/YashKikani24/asha-env](https://huggingface.co/spaces/YashKikani24/asha-env)

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Deployment**: Docker on HuggingFace Spaces
- **Constraints**: 2 vCPU / 8GB RAM, no GPU, port 7860
