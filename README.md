---
title: ASHA Agent Environment
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - healthcare
  - maternal-health
  - newborn-care
  - reinforcement-learning
---

# ASHA Agent Environment

A multi-step OpenEnv environment simulating **maternal and newborn care consultations** for ASHA (Accredited Social Health Activist) workers in rural India. The agent must elicit symptoms, order tests, diagnose, treat, or refer — with a limited kit, partial observability, and life-or-death stakes.

Built for the **Meta PyTorch × Scaler School of Technology OpenEnv Hackathon**.

🔗 **Live on Hugging Face Spaces**: [YashKikani24/asha-env](https://huggingface.co/spaces/YashKikani24/asha-env)

---

## What Makes This Different

Most medical AI benchmarks evaluate on static Q&A. This environment tests **active clinical decision-making**:

- **Partial observability**: The agent doesn't know the diagnosis — it must ask the right questions to uncover it
- **Resource constraints**: Only what's in the ASHA kit is available — no referral without reason
- **Stakes are asymmetric**: Missing an emergency (pre-eclampsia, birth asphyxia) is penalized far more than over-referring
- **Epidemiologically grounded**: Seasonal disease priors, regional boosts, and comorbidities shape each generated patient
- **15 conditions across maternal and newborn care** — from severe anaemia to neonatal sepsis

---

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

# Order a test from the ASHA kit
response = httpx.post(f"{BASE_URL}/step", json={
    "action": "order_test:haemoglobin_strip",
    "episode_id": episode_id
})

# Make a diagnosis (ends episode)
response = httpx.post(f"{BASE_URL}/step", json={
    "action": "diagnose:severe_anaemia",
    "episode_id": episode_id
})
result = response.json()
print(f"Reward: {result['reward']}")  # +1.0 if correct

# Reveal the hidden true diagnosis
state = httpx.get(f"{BASE_URL}/state").json()
print(f"True diagnosis: {state['true_diagnosis']}")
```

**No local setup needed** — point your client at the live Space:
```bash
BASE_URL=https://YashKikani24-asha-env.hf.space
```

---

## Building & Running

### Docker (Recommended)

```bash
docker build -t asha-env .
docker run -p 7860:7860 asha-env
curl http://localhost:7860/health
# {"status": "healthy", "service": "asha-env"}
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Running Tests

```bash
pytest tests/ -v                   # All 34 tests
pytest tests/test_env.py -v        # Environment logic
pytest tests/test_graders.py -v    # Grader scoring
pytest tests/test_patient_gen.py -v  # Patient generation
```

---

## Environment Details

### Episode Structure

Each episode is a **multi-step consultation**:

1. `POST /reset` — generates a patient (pregnant woman or newborn) with a hidden diagnosis
2. `POST /step` — agent takes actions (ask symptoms, order tests, treat, diagnose, refer)
3. Episode ends on **diagnosis**, **referral**, or hitting **max steps**
4. `GET /state` — reveals the true diagnosis and full episode state

### Tasks

| Task | Conditions | Max Steps | Key Challenge |
|------|-----------|-----------|---------------|
| `easy_diagnosis` | 5 | 5 | All symptoms visible, straightforward cases |
| `medium_consultation` | 10 | 12 | Partial observability, must ask questions |
| `hard_complex_case` | 15 | 20 | Comorbidities, emergencies, non-compliance |

### Actions

| Action Type | Format | Description |
|-------------|--------|-------------|
| Ask Symptom | `ask_symptom:pallor` | Elicit a specific symptom |
| Ask History | `ask_history:no_anc_visits` | Ask about patient or pregnancy history |
| Order Test | `order_test:bp_monitor` | Run a test from the ASHA kit |
| Treat | `treat:ifa_tablets` | Administer a medicine from the kit |
| Diagnose | `diagnose:pre_eclampsia` | Make a diagnosis (ends episode) |
| Refer | `refer:district_hospital` | Refer to a facility (ends episode) |

### Observation Fields

| Field | Description |
|-------|-------------|
| `patient.age`, `gender`, `chief_complaint` | Demographics (18–40 for mothers, 0 for newborns) |
| `patient.pregnant` | Whether the patient is currently pregnant |
| `patient.known_symptoms` | Symptoms revealed so far |
| `patient.vitals` | Temperature, BP, SpO2, pulse (+ weight_kg for newborns) |
| `patient.village_context` | Season, outbreaks, water source |
| `asha_context.kit_available` | Medicines currently in the ASHA kit |
| `asha_context.tests_available` | Tests available to order |
| `asha_context.patient_trust` | Trust level (affects compliance) |
| `available_actions` | Valid actions for the current state |
| `episode_id` | Episode identifier required for step requests |

---

## Reward

Step-level rewards guide learning:

| Condition | Reward |
|-----------|--------|
| Correct diagnosis | +1.0 |
| Same-category diagnosis | +0.3 |
| Wrong diagnosis | −0.3 |
| **Missed emergency diagnosis** | **−1.0** |
| Relevant symptom found | +0.05 × specificity |
| Negative symptom (rules out) | +0.01 |
| Relevant test ordered | +0.10 |
| Unnecessary test | −0.02 |
| Correct treatment | +0.2 |
| Wrong treatment | −0.1 |
| Correct referral | +0.8 |
| Over-referral (safe but wasteful) | +0.3 |
| **Under-referral on emergency** | **−1.0** |
| Invalid action | −0.1 |

---

## Conditions Covered

### Maternal (10)

| Condition | Severity | Emergency | Kit Treatment |
|-----------|----------|-----------|---------------|
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

| Condition | Severity | Emergency | Kit Treatment |
|-----------|----------|-----------|---------------|
| Birth Asphyxia | Critical | Yes | — (refer) |
| Neonatal Sepsis | Critical | Yes | — (refer) |
| Neonatal Jaundice | Medium | No | — (refer to PHC) |
| Low Birth Weight | High | No | ORS |
| Neonatal Hypothermia | High | No | — (refer to PHC) |

---

## ASHA Kit

| Category | Items |
|----------|-------|
| **Medicines** | IFA tablets, Calcium tablets, ORS, Zinc, Paracetamol, Misoprostol |
| **Tests** | BP monitor, Thermometer, Haemoglobin strip, Urine dipstick, Weighing scale |
| **Supplies** | Clean delivery kit, Gloves, Cord clamp, Mucus extractor, Blanket |

---

## Grading

The composite grader scores complete episodes on a 0.0–1.0 scale:

| Grader | Weight | What It Measures |
|--------|--------|------------------|
| **Diagnosis** | 40% | Correct diagnosis, category match, emergency detection |
| **Safety** | 25% | Emergency referral, harmful treatment avoidance |
| **Efficiency** | 20% | Questions asked vs. minimum needed, relevance |
| **Referral** | 15% | Correct referral level, over/under-referral penalties |

---

## Baseline Scores

| Agent | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Random | ~0.48 | ~0.47 | ~0.45 | ~0.47 |
| Rule-based | ~0.88 | ~0.59 | ~0.64 | ~0.70 |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode (optional `task_id`) |
| POST | `/step` | Take action (`action` + `episode_id`) |
| GET | `/state` | Full state including true diagnosis |
| GET | `/health` | Health check |

---

## Project Structure

```
metahak/
├── asha_env/                  # Core environment module
│   ├── env.py                 # AshaEnv class (Gymnasium interface)
│   ├── data/
│   │   ├── diseases.json      # 15 maternal/newborn conditions
│   │   ├── symptoms.json      # 62 symptoms (maternal + newborn)
│   │   ├── asha_kit.json      # Medicines, tests, supplies
│   │   └── villages.json      # Village contexts with regional metadata
│   ├── patient/
│   │   ├── generator.py       # Generates pregnant women or newborns
│   │   ├── epidemiology.py    # Seasonal priors, regional disease boosts
│   │   └── comorbidity.py     # Co-occurring condition logic
│   └── tasks/
│       ├── easy_task.py       # 5 conditions, 5 steps
│       ├── medium_task.py     # 10 conditions, 12 steps
│       └── hard_task.py       # 15 conditions, 20 steps
├── graders/
│   ├── composite_grader.py    # Weighted blend of 4 sub-graders
│   ├── diagnosis_grader.py    # 40% — diagnostic accuracy
│   ├── safety_grader.py       # 25% — emergency & safety
│   ├── efficiency_grader.py   # 20% — consultation efficiency
│   └── referral_grader.py     # 15% — referral appropriateness
├── baseline/
│   ├── random_agent.py        # Uniform random actions
│   └── rule_based_agent.py    # Handcrafted clinical rules
├── server/
│   └── app.py                 # FastAPI server (4 endpoints)
├── tests/                     # 34 pytest tests
├── inference.py               # LLM inference script
├── Dockerfile
├── requirements.txt
├── openenv.yaml               # OpenEnv manifest
└── reward.md                  # Detailed reward function reference
```

---

## Deployment Specs

| | Value |
|--|-------|
| **Compute** | 2 vCPU / 8GB RAM, no GPU required |
| **Port** | 7860 |
| **Framework** | FastAPI + Uvicorn |
| **Interface** | Gymnasium-compatible |
| **Deployment** | Docker on HuggingFace Spaces |

---

## Limitations

- **No multi-agent support**: Single patient per episode; no concurrent consultation simulation
- **Static complication progression**: Conditions don't evolve within a single episode (e.g., pre-eclampsia can't worsen to eclampsia mid-consultation)
- **Kit is fixed**: No mechanism to request additional supplies from a PHC mid-episode
- **Text-only observation**: No vitals trend charts or visual patient data

---

## Use Cases

- **LLM Evaluation**: Benchmark clinical reasoning on maternal/newborn health scenarios
- **RL Agent Training**: Multi-step diagnostic environments with shaped reward signals
- **ASHA Worker Training**: Simulation for community health worker decision-making education
- **Research**: Reproducible environments with seasonal variation, regional epidemiology, and comorbidity modeling

---

## Tech Stack

FastAPI · Uvicorn · Gymnasium · Docker · HuggingFace Spaces · OpenEnv

---

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [ASHA Programme — NHM India](https://nhm.gov.in/index1.php?lang=1&level=1&sublinkid=150&lid=226)
- [Meta PyTorch × Scaler OpenEnv Hackathon](https://huggingface.co/meta-pytorch)

---

## License

BSD-3-Clause License (see [OpenEnv LICENSE](https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE))
