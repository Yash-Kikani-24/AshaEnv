---
title: ASHA Agent Environment
emoji: "\U0001F3E5"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - healthcare
  - maternal-health
  - newborn-care
---

# ASHA Agent Environment

An OpenEnv environment that simulates **maternal and newborn care** consultations for ASHA (Accredited Social Health Activist) workers in rural India. Each episode presents a pregnant woman or newborn with danger signs, and the agent must ask symptoms, order tests, diagnose, treat, or refer — all within a limited number of steps using a realistic ASHA medical kit.

Built for the **Meta PyTorch x Scaler School of Technology OpenEnv Hackathon**.

## What Problem Does This Solve?

Over 600 million people in rural India depend on a network of ~1 million ASHA workers as their first point of healthcare contact. These community health workers are especially critical for **maternal and newborn health** — conducting antenatal checkups, identifying pregnancy danger signs, assisting safe deliveries, and recognizing neonatal emergencies.

ASHAs operate with limited medical kits and must make critical decisions: Is this headache just a headache, or a sign of pre-eclampsia? Should this newborn with jaundice be treated at home or referred urgently? When a mother is bleeding after delivery, is misoprostol appropriate or should she be rushed to the district hospital?

This environment lets AI agents practice these exact decisions in a realistic simulation with **15 maternal/newborn conditions**, seasonal disease patterns, regional epidemiology, comorbidities, patient non-compliance, and resource constraints — creating a safe space to learn clinical decision-making before any real-world deployment.

## Quick Start

```python
import httpx

BASE_URL = "http://localhost:7860"

# Start a new episode
response = httpx.post(f"{BASE_URL}/reset", json={"task_id": "medium_consultation"})
data = response.json()
episode_id = data["episode_id"]
print(f"Patient: {data['patient']['chief_complaint']}")
# Patient: pallor, fatigue for 3 days

# Ask a symptom
response = httpx.post(f"{BASE_URL}/step", json={
    "action": "ask_symptom:weakness",
    "episode_id": episode_id
})
result = response.json()
print(f"Reward: {result['reward']}, Done: {result['done']}")

# Order a test
response = httpx.post(f"{BASE_URL}/step", json={
    "action": "order_test:haemoglobin_strip",
    "episode_id": episode_id
})

# Make a diagnosis
response = httpx.post(f"{BASE_URL}/step", json={
    "action": "diagnose:severe_anaemia",
    "episode_id": episode_id
})
result = response.json()
print(f"Reward: {result['reward']}")  # +1.0 if correct

# Check the true diagnosis
state = httpx.get(f"{BASE_URL}/state").json()
print(f"True diagnosis: {state['true_diagnosis']}")
```

## Building & Running

### Docker (Recommended)

```bash
docker build -t asha-env .
docker run -p 7860:7860 asha-env
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Running Tests

```bash
# All 34 tests
pytest tests/ -v

# Individual test files
pytest tests/test_env.py -v
pytest tests/test_graders.py -v
pytest tests/test_patient_gen.py -v
```

## Deploying to Hugging Face Spaces

```bash
# From the project directory (where openenv.yaml is located)
openenv push

# Or with options
openenv push --namespace my-org --private
```

Live at: [https://huggingface.co/spaces/YashKikani24/asha-env](https://huggingface.co/spaces/YashKikani24/asha-env)

## Environment Details

### Episode Structure

Each episode is a **multi-step consultation**:

1. `POST /reset` — generates a patient (pregnant woman or newborn) with a hidden diagnosis
2. `POST /step` — agent takes actions (ask symptoms, order tests, treat, etc.)
3. Episode ends when the agent makes a **diagnosis** or **referral**, or hits **max steps**
4. `GET /state` — reveals the true diagnosis for evaluation

### Actions

| Action Type | Format | Description |
|-------------|--------|-------------|
| Ask Symptom | `ask_symptom:pallor` | Ask if the patient has a specific symptom |
| Ask History | `ask_history:no_anc_visits` | Ask about patient/pregnancy history |
| Order Test | `order_test:bp_monitor` | Run a test from the ASHA kit |
| Diagnose | `diagnose:pre_eclampsia` | Make a diagnosis (ends episode) |
| Treat | `treat:ifa_tablets` | Administer medicine from kit |
| Refer | `refer:district_hospital` | Refer to facility (ends episode) |

### Observation

| Field | Description |
|-------|-------------|
| `patient.age`, `gender`, `chief_complaint` | Demographics (age 18-40 for mothers, 0 for newborns) |
| `patient.pregnant` | Whether the patient is pregnant |
| `patient.known_symptoms` | Symptoms revealed so far |
| `patient.vitals` | Temperature, BP, SpO2, pulse (+ weight_kg for newborns) |
| `patient.village_context` | Season, outbreaks, water source |
| `asha_context.kit_available` | Medicines in the ASHA kit |
| `asha_context.tests_available` | Tests available |
| `asha_context.patient_trust` | Trust level (affects compliance) |
| `available_actions` | Valid actions for current state |
| `episode_id` | Episode identifier for step requests |

### Reward

Step-level rewards guide learning:

| Condition | Reward |
|-----------|--------|
| Correct diagnosis | +1.0 |
| Same-category diagnosis | +0.3 |
| Wrong diagnosis | -0.3 |
| Missed emergency diagnosis | -1.0 |
| Relevant symptom found | +0.05 × specificity |
| Negative symptom finding | +0.01 |
| Relevant test ordered | +0.10 |
| Unnecessary test | -0.02 |
| Correct treatment | +0.2 |
| Wrong treatment | -0.1 |
| Correct referral | +0.8 |
| Over-referral (safe but wasteful) | +0.3 |
| Under-referral on emergency | -1.0 |
| Invalid action | -0.1 |

## Tasks

| Task | Conditions | Max Steps | Key Challenge |
|------|------------|-----------|---------------|
| `easy_diagnosis` | 5 | 5 | All symptoms visible, straightforward cases |
| `medium_consultation` | 10 | 12 | Partial observability, must ask questions |
| `hard_complex_case` | 15 | 20 | Comorbidities, emergencies, non-compliance |

## Conditions Covered

### Maternal (10)

| Condition | Severity | Emergency | ASHA Kit Treatment |
|-----------|----------|-----------|-------------------|
| Severe Anaemia in Pregnancy | High | No | IFA tablets |
| Pre-eclampsia | Critical | Yes | — (refer) |
| Eclampsia | Critical | Yes | — (refer) |
| Antepartum Haemorrhage | Critical | Yes | — (refer) |
| Postpartum Haemorrhage | Critical | Yes | Misoprostol |
| Puerperal Sepsis | High | No | Paracetamol |
| Gestational Diabetes | High | No | — (refer) |
| Hyperemesis Gravidarum | Medium | No | ORS |
| Preterm Labour | Critical | Yes | — (refer) |
| Obstructed Labour | Critical | Yes | — (refer) |

### Newborn (5)

| Condition | Severity | Emergency | ASHA Kit Treatment |
|-----------|----------|-----------|-------------------|
| Birth Asphyxia | Critical | Yes | — (refer) |
| Neonatal Sepsis | Critical | Yes | — (refer) |
| Neonatal Jaundice | Medium | No | — (refer to PHC) |
| Low Birth Weight | High | No | ORS |
| Neonatal Hypothermia | High | No | — (refer to PHC) |

## ASHA Kit

**Medicines**: IFA tablets, Calcium tablets, ORS, Zinc, Paracetamol, Misoprostol

**Tests**: BP monitor, Thermometer, Haemoglobin strip, Urine dipstick, Weighing scale

**Supplies**: Clean delivery kit, Gloves, Cord clamp, Mucus extractor, Blanket

## Grading

The composite grader scores complete episodes on a 0.0 to 1.0 scale:

| Grader | Weight | What It Measures |
|--------|--------|------------------|
| **Diagnosis** | 40% | Correct diagnosis, category match, emergency detection |
| **Safety** | 25% | Emergency referral, harmful treatment avoidance |
| **Efficiency** | 20% | Questions asked vs. minimum needed, relevance |
| **Referral** | 15% | Correct referral level, over/under-referral penalties |

## Baseline Scores

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | ~0.48 | ~0.47 | ~0.45 | ~0.47 |
| Rule-based | ~0.88 | ~0.59 | ~0.64 | ~0.70 |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode (optional `task_id`) |
| POST | `/step` | Take action (`action` + `episode_id`) |
| GET | `/state` | Full state including true diagnosis |
| GET | `/health` | Health check |

## Project Structure

```
metahak/
├── asha_env/                  # Core environment module
│   ├── env.py                 # AshaEnv class (Gymnasium interface)
│   ├── data/                  # JSON data files
│   │   ├── diseases.json      # 15 maternal/newborn conditions
│   │   ├── symptoms.json      # 62 symptoms (maternal + newborn)
│   │   ├── asha_kit.json      # Medicines, tests, supplies
│   │   └── villages.json      # Village contexts with regions
│   ├── patient/               # Patient generation
│   │   ├── generator.py       # Generates pregnant women or newborns
│   │   ├── epidemiology.py    # Seasonal priors, regional boosts
│   │   └── comorbidity.py     # Co-occurring condition logic
│   └── tasks/                 # Difficulty configurations
│       ├── easy_task.py       # 5 conditions, 5 steps
│       ├── medium_task.py     # 10 conditions, 12 steps
│       └── hard_task.py       # 15 conditions, 20 steps
├── graders/                   # Episode scoring system
│   ├── composite_grader.py    # Weighted blend of 4 sub-graders
│   ├── diagnosis_grader.py    # 40% — diagnostic accuracy
│   ├── safety_grader.py       # 25% — emergency & safety
│   ├── efficiency_grader.py   # 20% — consultation efficiency
│   └── referral_grader.py     # 15% — referral appropriateness
├── baseline/                  # Benchmark agents
│   ├── random_agent.py        # Uniform random actions
│   └── rule_based_agent.py    # Handcrafted clinical rules
├── server/
│   └── app.py                 # FastAPI server (4 endpoints)
├── tests/                     # 34 pytest tests
├── inference.py               # LLM inference script
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
├── openenv.yaml               # OpenEnv manifest
└── reward.md                  # Detailed reward function reference
```

## Use Cases

- **LLM Evaluation**: Benchmark clinical reasoning for maternal/newborn health scenarios
- **Agent Training**: Train RL agents on multi-step diagnostic consultations with realistic reward signals
- **ASHA Worker Training**: Simulate decision-making scenarios for community health worker education
- **Research**: Reproducible healthcare environments with seasonal variation, regional epidemiology, and comorbidity modeling

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Environment**: Gymnasium-compatible interface
- **Deployment**: Docker on HuggingFace Spaces
- **Constraints**: 2 vCPU / 8GB RAM, no GPU, port 7860

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [ASHA Programme (NHM India)](https://nhm.gov.in/index1.php?lang=1&level=1&sublinkid=150&lid=226)
