"""
ASHA Environment — core RL (Reinforcement Learning) environment.

Simulates a rural Indian ASHA (Accredited Social Health Activist) worker
diagnosing and managing maternal and newborn health conditions.

The agent (LLM or policy) interacts via a step-based loop:
  1. reset(task_id) → generates a new patient episode and returns an observation.
  2. step(action)   → processes one action, returns (observation, reward, done, info).

Actions are strings like "ask_symptom:pallor", "diagnose:severe_anaemia",
"treat:ifa_tablets", "refer:phc", etc.

Rewards encourage:
  - Asking relevant symptoms/history  (+small positive)
  - Ordering useful tests              (+0.10)
  - Correct diagnosis                  (+1.0)
  - Correct referral level             (+0.8)
  - Correct treatment                  (+0.2)
Penalties apply for wrong/unnecessary actions, missed emergencies, etc.
"""

import json
import os
import random
import uuid

from .patient.generator import generate_patient
from .patient.epidemiology import get_season, load_diseases
from .tasks.easy_task import EasyTask
from .tasks.medium_task import MediumTask
from .tasks.hard_task import HardTask

# Path to the JSON data files (diseases, symptoms, ASHA kit, villages)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Maps task_id strings (used in reset()) to their task configuration classes.
# Each task class defines difficulty, allowed diseases, max steps, etc.
TASK_MAP = {
    "easy_diagnosis": EasyTask,
    "medium_consultation": MediumTask,
    "hard_complex_case": HardTask,
}

# Ordered from least to most urgent. Used to score referral decisions —
# under-referring an emergency is heavily penalized (-1.0).
REFERRAL_LEVELS = ["none", "phc", "district_hospital", "emergency"]


class AshaEnv:
    """
    Main environment class. One instance can run many episodes sequentially.

    Lifecycle:
        env = AshaEnv()
        obs = env.reset("medium_consultation")   # start a new patient episode
        while not done:
            obs, reward, done, info = env.step("ask_symptom:pallor")
        state = env.get_state()                   # ground-truth for grading
    """

    def __init__(self):
        self._load_data()

        # --- Episode state (all reset on each reset() call) ---
        self.episode_id = None          # Unique ID for the current episode
        self.patient = None             # Dict holding all patient data (see generator.py)
        self.task = None                # Task class (EasyTask / MediumTask / HardTask)
        self.step_count = 0             # Number of actions taken so far
        self.done = False               # Whether the episode has ended
        self.trajectory = []            # Ordered list of all actions taken
        self.asked_symptoms = []        # Symptom IDs already asked about
        self.asked_history = []         # History items already asked about
        self.ordered_tests = []         # Tests already ordered
        self.diagnosis_made = None      # The disease_id the agent diagnosed (or None)
        self.referral_made = None       # The referral level the agent chose (or None)
        self.treatments_given = []      # Medicines administered during the episode
        self.total_reward = 0.0         # Cumulative reward for the episode

    def _load_data(self):
        """Load all static reference data from JSON files into memory."""
        with open(os.path.join(DATA_DIR, "diseases.json")) as f:
            self.diseases_db = {d["id"]: d for d in json.load(f)}
        with open(os.path.join(DATA_DIR, "symptoms.json")) as f:
            self.symptoms_db = json.load(f)
        with open(os.path.join(DATA_DIR, "asha_kit.json")) as f:
            self.kit = json.load(f)
        with open(os.path.join(DATA_DIR, "villages.json")) as f:
            self.villages = json.load(f)

    def reset(self, task_id: str = "medium_consultation") -> dict:
        """
        Start a new episode. Generates a fresh patient and returns the initial observation.

        Args:
            task_id: One of "easy_diagnosis", "medium_consultation", "hard_complex_case".
                     Controls which diseases can appear, max steps, difficulty, etc.

        Returns:
            observation dict visible to the agent (patient info, available actions, etc.).
        """
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
        """
        Execute one action and advance the episode.

        Args:
            action: A string like "ask_symptom:pallor", "diagnose:severe_anaemia",
                    "treat:ifa_tablets", "refer:phc", "order_test:bp_monitor", etc.

        Returns:
            (observation, reward, done, info) tuple — standard RL env interface.
            - observation: updated patient/context dict visible to the agent.
            - reward: float reward for this action (can be negative).
            - done: True if episode has ended (max steps reached or terminal action).
            - info: dict with message, validity flag, step count, cumulative reward.
        """
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
        """
        Return the full ground-truth state of the episode (for grading/evaluation).
        Unlike _build_observation(), this exposes hidden info like the true diagnosis,
        all true symptoms, comorbidities, and the complete patient dict.
        """
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
        """Route a parsed action to the appropriate handler and return reward."""
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
        """
        Ask the patient about a specific symptom.
        - On hard difficulty, the patient may be non-compliant (give unreliable answers).
        - Finding a true symptom rewards proportionally to that symptom's specificity.
        - Asking about a symptom the patient doesn't have gives a small +0.01
          (negative findings are still useful clinically).
        """
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
        """
        Ask about a patient's medical/social history item (e.g., "poor_diet", "home_delivery").
        Returns +0.05 if the item is present in the patient's history, +0.01 otherwise.
        """
        self.asked_history.append(history_item)

        if history_item in self.patient["history"]:
            if history_item not in self.patient["revealed_history"]:
                self.patient["revealed_history"].append(history_item)
            return 0.05
        return 0.01

    def _order_test(self, test_id: str) -> float:
        """
        Order a point-of-care test from the ASHA kit (e.g., bp_monitor, thermometer).
        Rewards +0.10 if the test is clinically relevant for the patient's true disease,
        otherwise penalizes -0.02 for an unnecessary test. Also adjusts patient trust.
        """
        self.ordered_tests.append(test_id)
        disease = self.patient["disease_data"]
        true_diag = self.patient["true_diagnosis"]

        # Maps each test to the diseases it is clinically useful for
        test_relevance = {
            "bp_monitor": ["pre_eclampsia", "eclampsia", "postpartum_haemorrhage", "antepartum_haemorrhage"],
            "thermometer": ["puerperal_sepsis", "neonatal_sepsis", "hypothermia_newborn"],
            "haemoglobin_strip": ["severe_anaemia", "postpartum_haemorrhage"],
            "urine_dipstick": ["gestational_diabetes", "pre_eclampsia", "puerperal_sepsis"],
            "weighing_scale": ["low_birth_weight", "hyperemesis", "neonatal_jaundice"],
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
        """
        Make a final diagnosis. This is a TERMINAL action — ends the episode.

        Scoring:
          +1.0  exact correct diagnosis
          +0.3  wrong disease but same category (e.g., both maternal)
          -1.0  missed an emergency condition (dangerous!)
          -0.3  wrong diagnosis otherwise
        """
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
        """
        Administer a medicine from the ASHA kit.
          +0.2  correct treatment for the disease
          -0.1  wrong medicine but at least it's in the kit
          -0.2  medicine not even in the kit (completely invalid)
        """
        self.treatments_given.append(medicine_id)
        disease = self.patient["disease_data"]

        if medicine_id in disease.get("kit_treatment", []):
            return 0.2
        elif medicine_id in self.kit["medicines"]:
            # Available but wrong medicine
            return -0.1
        return -0.2

    def _refer(self, level: str) -> float:
        """
        Refer the patient to a higher facility. This is a TERMINAL action.

        Levels (lowest → highest): none → phc → district_hospital → emergency.

        Scoring:
          +0.8  correct referral level
          +0.3  over-referred (safe but wastes resources)
          -0.3  under-referred
          -1.0  under-referred AND the disease is an emergency (life-threatening mistake)
        """
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
        """
        Build the observation dict that the agent sees each step.

        Contains only information the ASHA worker would realistically know:
        - Patient demographics, chief complaint, revealed symptoms, vitals
        - Village context (season, outbreaks, water source)
        - ASHA context (kit contents, steps taken, patient trust level)
        - List of all currently available actions

        Does NOT expose: true diagnosis, hidden symptoms, or disease data.
        """
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
        """
        Generate the full list of valid action strings the agent can take right now.
        Actions already taken (symptoms asked, tests ordered) are excluded.
        Includes: ask_symptom, ask_history, order_test, diagnose, treat, refer.
        """
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
            "poor_diet", "no_ifa_supplements", "multiple_pregnancies",
            "vegetarian_diet", "first_pregnancy", "family_history_bp",
            "previous_pre_eclampsia", "teen_pregnancy", "age_over_35",
            "no_anc_visits", "previous_caesarean", "placenta_previa_history",
            "smoking_in_household", "grand_multipara", "prolonged_labour_history",
            "previous_pph", "anaemia_in_pregnancy", "home_delivery",
            "premature_rupture_membranes", "no_clean_delivery_kit",
            "family_history_diabetes", "previous_gdm", "obesity",
            "previous_large_baby", "multiple_gestation", "previous_hyperemesis",
            "previous_preterm", "cervical_incompetence", "uti_in_pregnancy",
            "short_stature", "no_skilled_attendant", "premature_birth",
            "maternal_fever_during_labour", "unclean_cord_care",
            "blood_group_incompatibility", "breastfeeding_difficulty",
            "low_birth_weight_history", "cold_environment",
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
