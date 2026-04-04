import json
import os
import random
import uuid

from .patient.generator import generate_patient
from .patient.epidemiology import get_season, load_diseases
from .tasks.easy_task import EasyTask
from .tasks.medium_task import MediumTask
from .tasks.hard_task import HardTask

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

TASK_MAP = {
    "easy_diagnosis": EasyTask,
    "medium_consultation": MediumTask,
    "hard_complex_case": HardTask,
}

REFERRAL_LEVELS = ["none", "phc", "district_hospital", "emergency"]


class AshaEnv:
    def __init__(self):
        self._load_data()
        self.episode_id = None
        self.patient = None
        self.task = None
        self.step_count = 0
        self.done = False
        self.trajectory = []
        self.asked_symptoms = []
        self.asked_history = []
        self.ordered_tests = []
        self.diagnosis_made = None
        self.referral_made = None
        self.treatments_given = []
        self.total_reward = 0.0

    def _load_data(self):
        with open(os.path.join(DATA_DIR, "diseases.json")) as f:
            self.diseases_db = {d["id"]: d for d in json.load(f)}
        with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
            self.symptoms_db = json.load(f)
        with open(os.path.join(DATA_DIR, "asha_kit.json")) as f:
            self.kit = json.load(f)
        with open(os.path.join(DATA_DIR, "villages.json")) as f:
            self.villages = json.load(f)

    def reset(self, task_id: str = "medium_consultation") -> dict:
        task_cls = TASK_MAP.get(task_id)
        if task_cls is None:
            raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(TASK_MAP.keys())}")

        self.task = task_cls
        self.episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        self.step_count = 0
        self.done = False
        self.trajectory = []
        self.asked_symptoms = []
        self.asked_history = []
        self.ordered_tests = []
        self.diagnosis_made = None
        self.referral_made = None
        self.treatments_given = []
        self.total_reward = 0.0

        # Pick a village
        village = random.choice(self.villages)
        season = get_season()

        # Generate patient
        self.patient = generate_patient(
            task_difficulty=task_cls.difficulty,
            season=season,
            village=village,
            allowed_diseases=task_cls.diseases,
        )

        return self._build_observation()

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if self.done:
            return self._build_observation(), 0.0, True, {"message": "Episode already ended."}

        self.step_count += 1
        self.trajectory.append(action)

        # Parse action
        parts = action.split(":", 1)
        action_type = parts[0]
        action_value = parts[1] if len(parts) > 1 else ""

        # Validate action
        available = self._get_available_actions()
        if action not in available:
            reward = -0.1
            self.total_reward += reward
            info = {"message": f"Invalid action: {action}", "valid": False}
            if self.step_count >= self.task.max_steps:
                self.done = True
                info["message"] += " Episode ended (max steps)."
            return self._build_observation(), reward, self.done, info

        # Process action
        reward = self._process_action(action_type, action_value)
        self.total_reward += reward

        # Check if episode should end
        if self.step_count >= self.task.max_steps:
            self.done = True

        # Terminal actions end the episode
        if action_type in ("diagnose", "refer"):
            self.done = True

        info = {
            "message": "OK",
            "valid": True,
            "step": self.step_count,
            "total_reward": round(self.total_reward, 3),
        }

        return self._build_observation(), round(reward, 3), self.done, info

    def get_state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "true_diagnosis": self.patient["true_diagnosis"] if self.patient else None,
            "comorbidities": self.patient["comorbidities"] if self.patient else [],
            "true_symptoms": self.patient["true_symptoms"] if self.patient else [],
            "history": self.patient["history"] if self.patient else [],
            "trajectory": self.trajectory,
            "step_count": self.step_count,
            "total_reward": round(self.total_reward, 3),
            "done": self.done,
            "diagnosis_made": self.diagnosis_made,
            "referral_made": self.referral_made,
            "treatments_given": self.treatments_given,
            "patient": self.patient,
        }

    def _process_action(self, action_type: str, action_value: str) -> float:
        if action_type == "ask_symptom":
            return self._ask_symptom(action_value)
        elif action_type == "ask_history":
            return self._ask_history(action_value)
        elif action_type == "order_test":
            return self._order_test(action_value)
        elif action_type == "diagnose":
            return self._diagnose(action_value)
        elif action_type == "treat":
            return self._treat(action_value)
        elif action_type == "refer":
            return self._refer(action_value)
        return -0.1

    def _ask_symptom(self, symptom_id: str) -> float:
        self.asked_symptoms.append(symptom_id)

        # Non-compliance: patient may not answer truthfully on hard
        if random.random() < self.patient["non_compliance_rate"]:
            # Patient gives unreliable answer — don't reveal
            self.patient["patient_trust"] = max(0, self.patient["patient_trust"] - 2)
            return -0.02

        # Check if patient has this symptom
        if symptom_id in self.patient["true_symptoms"]:
            if symptom_id not in self.patient["revealed_symptoms"]:
                self.patient["revealed_symptoms"].append(symptom_id)
            # Reward for finding relevant symptom
            specificity = self.symptoms_db.get(symptom_id, {}).get("specificity", 0.5)
            return 0.05 * specificity
        else:
            # Negative finding — still useful but less rewarding
            return 0.01

    def _ask_history(self, history_item: str) -> float:
        self.asked_history.append(history_item)

        if history_item in self.patient["history"]:
            if history_item not in self.patient["revealed_history"]:
                self.patient["revealed_history"].append(history_item)
            return 0.05
        return 0.01

    def _order_test(self, test_id: str) -> float:
        self.ordered_tests.append(test_id)
        disease = self.patient["disease_data"]
        true_diag = self.patient["true_diagnosis"]

        # Simulate test results
        test_relevance = {
            "malaria_rdt": ["malaria"],
            "haemoglobin_strip": ["anaemia", "malnutrition"],
            "urine_dipstick": ["diabetes", "pre_eclampsia"],
            "bp_monitor": ["hypertension", "pre_eclampsia"],
            "thermometer": ["malaria", "dengue", "typhoid", "pneumonia", "ari", "chickenpox"],
        }

        relevant_diseases = test_relevance.get(test_id, [])
        if true_diag in relevant_diseases:
            # Test is relevant — good choice
            self.patient["patient_trust"] = min(100, self.patient["patient_trust"] + 3)
            return 0.10
        else:
            # Unnecessary test
            return -0.02

    def _diagnose(self, disease_id: str) -> float:
        self.diagnosis_made = disease_id
        true_diag = self.patient["true_diagnosis"]
        true_disease = self.diseases_db[true_diag]

        if disease_id == true_diag:
            # Correct diagnosis
            return 1.0
        elif self.diseases_db.get(disease_id, {}).get("category") == true_disease["category"]:
            # Same category
            return 0.3
        elif true_disease.get("emergency"):
            # Missed an emergency — dangerous
            return -1.0
        else:
            return -0.3

    def _treat(self, medicine_id: str) -> float:
        self.treatments_given.append(medicine_id)
        disease = self.patient["disease_data"]

        if medicine_id in disease.get("kit_treatment", []):
            return 0.2
        elif medicine_id in self.kit["medicines"]:
            # Available but wrong medicine
            return -0.1
        return -0.2

    def _refer(self, level: str) -> float:
        self.referral_made = level
        disease = self.patient["disease_data"]
        correct_level = disease["referral_level"]

        correct_idx = REFERRAL_LEVELS.index(correct_level) if correct_level in REFERRAL_LEVELS else 1
        given_idx = REFERRAL_LEVELS.index(level) if level in REFERRAL_LEVELS else 1

        if level == correct_level:
            return 0.8
        elif given_idx > correct_idx:
            # Over-referred
            return 0.3
        else:
            # Under-referred
            if disease.get("emergency"):
                return -1.0
            return -0.3

    def _build_observation(self) -> dict:
        if self.patient is None:
            return {}

        p = self.patient
        demo = p["demographics"]
        village = p["village"]

        return {
            "patient": {
                "age": demo["age"],
                "gender": demo["gender"],
                "chief_complaint": p["chief_complaint"],
                "location": village["region"],
                "socioeconomic": demo["socioeconomic"],
                "pregnant": demo["pregnant"],
                "known_symptoms": list(p["revealed_symptoms"]),
                "vitals": p["vitals"],
                "revealed_history": list(p["revealed_history"]),
                "village_context": {
                    "season": p["season"],
                    "active_outbreaks": village.get("active_outbreaks", []),
                    "recent_cases": village.get("recent_cases", []),
                    "water_source": village.get("water_source", "unknown"),
                },
            },
            "asha_context": {
                "season": p["season"],
                "kit_available": self.kit["medicines"],
                "tests_available": self.kit["tests"],
                "patient_trust": p["patient_trust"],
                "steps_taken": self.step_count,
                "max_steps": self.task.max_steps,
                "visit_number": 1,
            },
            "available_actions": self._get_available_actions(),
            "episode_id": self.episode_id,
        }

    def _get_available_actions(self) -> list[str]:
        if self.done:
            return []

        actions = []
        true_diag = self.patient["true_diagnosis"]

        # Symptoms not yet asked
        all_symptom_ids = list(self.symptoms_db.keys())
        for sid in all_symptom_ids:
            if sid not in self.asked_symptoms:
                actions.append(f"ask_symptom:{sid}")

        # History items not yet asked
        all_history = [
            "recent_travel", "mosquito_exposure", "previous_malaria",
            "neighbourhood_cases", "contaminated_water", "street_food",
            "tb_contact", "previous_tb", "overcrowded_living",
            "poor_diet", "heavy_menstruation", "vegetarian_diet",
            "smoking", "recent_cold", "no_handwashing",
            "food_insecurity", "family_history_bp", "salt_heavy_diet",
            "stress", "family_history_diabetes", "sedentary_lifestyle",
            "obesity", "school_exposure", "no_vaccination",
            "alcohol_use", "previous_hepatitis", "barefoot_walking",
            "no_deworming", "poor_hygiene", "first_pregnancy",
            "previous_pre_eclampsia", "cold_exposure", "smoking_in_household",
        ]
        for h in all_history:
            if h not in self.asked_history:
                actions.append(f"ask_history:{h}")

        # Tests not yet ordered
        for test in self.kit["tests"]:
            if test not in self.ordered_tests:
                actions.append(f"order_test:{test}")

        # Diagnose options — all diseases for this task
        for did in self.task.diseases:
            actions.append(f"diagnose:{did}")

        # Treatment options
        for med in self.kit["medicines"]:
            actions.append(f"treat:{med}")

        # Referral options
        for level in REFERRAL_LEVELS:
            actions.append(f"refer:{level}")

        return actions
