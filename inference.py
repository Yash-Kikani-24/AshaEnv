"""
inference.py — ASHA Agent Environment inference script.

Uses an LLM to play episodes against the ASHA environment over HTTP.
Environment URL: https://YashKikani24-asha-env.hf.space (or local via API_BASE_URL)

Required env vars:
  API_BASE_URL  — base URL of the environment server (e.g. http://localhost:7860)
  HF_TOKEN      — HuggingFace token or API key for the LLM
  MODEL_NAME    — model name to use for inference
"""

import os
import re
import sys
import json
import httpx

# --- Config ---
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", ""))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

SYSTEM_PROMPT = """You are an ASHA (Accredited Social Health Activist) worker in rural India.
You are conducting a patient consultation. Your job is to:
1. Ask relevant symptom questions to narrow down the diagnosis
2. Order appropriate tests from your ASHA kit when needed
3. Make a diagnosis when you have enough information
4. Treat with available medicines or refer to the appropriate facility

You have a limited medical kit with: ORS, iron tablets, OCP, chloroquine, paracetamol, zinc, albendazole.
You can run tests: malaria RDT, haemoglobin strip, urine dipstick, BP monitor, thermometer.

IMPORTANT: You must respond with EXACTLY ONE action from the available_actions list.
Respond with just the action string, nothing else. Example: ask_symptom:fever

Be efficient — ask the most informative questions first. Don't waste steps on unlikely symptoms.
If you suspect an emergency (e.g. pre-eclampsia, severe dehydration), refer immediately."""

TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]
EPISODES_PER_TASK = 5


def parse_action(raw_text: str, available_actions: list[str]) -> str:
    """Parse an action from LLM output, with regex fallback."""
    raw = raw_text.strip()

    # Direct match
    if raw in available_actions:
        return raw

    # Try to find an action pattern in the text
    # Patterns: "action_type:value"
    patterns = [
        r'(ask_symptom:\w+)',
        r'(ask_history:\w+)',
        r'(order_test:\w+)',
        r'(diagnose:\w+)',
        r'(treat:\w+)',
        r'(refer:\w+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            candidate = match.group(1)
            if candidate in available_actions:
                return candidate

    # Try lowercase match
    raw_lower = raw.lower().strip()
    for action in available_actions:
        if action.lower() == raw_lower:
            return action

    # Try substring match — find any available action mentioned in the text
    for action in available_actions:
        if action in raw:
            return action

    # Fallback: first available action
    if available_actions:
        return available_actions[0]

    return raw


def call_llm(messages: list[dict]) -> str:
    """Call the LLM API and return the response text."""
    try:
        # Try OpenAI-compatible API
        api_url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.3,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""


def run_episode(task_id: str, use_llm: bool = True) -> dict:
    """Run a single episode against the environment."""
    with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
        # Reset
        resp = client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
        episode_id = obs["episode_id"]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            available = obs.get("available_actions", [])
            if not available:
                break

            if use_llm:
                # Build user message with observation
                user_msg = (
                    f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
                    f"Choose ONE action from available_actions. Respond with just the action string."
                )
                messages.append({"role": "user", "content": user_msg})
                raw_response = call_llm(messages)
                action = parse_action(raw_response, available)
                messages.append({"role": "assistant", "content": action})
            else:
                # Fallback: pick first available action
                action = available[0]

            # Step
            resp = client.post("/step", json={"action": action, "episode_id": episode_id})
            resp.raise_for_status()
            step_data = resp.json()

            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward
            steps += 1

            print(f"    Step {steps}: {action} -> reward={reward:.3f}")

        # Get final state for grading
        resp = client.get("/state")
        state = resp.json()

        return {
            "task_id": task_id,
            "episode_id": episode_id,
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "true_diagnosis": state.get("true_diagnosis"),
            "diagnosis_made": state.get("diagnosis_made"),
            "referral_made": state.get("referral_made"),
        }


def main():
    use_llm = HF_TOKEN and MODEL_NAME and MODEL_NAME != "test"
    if not use_llm:
        print("No LLM configured (HF_TOKEN/MODEL_NAME missing or MODEL_NAME=test).")
        print("Running in fallback mode (first available action).\n")

    results = {}
    all_scores = []

    for task_id in TASKS:
        print(f"\n{'='*50}")
        print(f"Task: {task_id}")
        print(f"{'='*50}")

        task_scores = []
        for ep in range(EPISODES_PER_TASK):
            print(f"\n  Episode {ep+1}/{EPISODES_PER_TASK}:")
            result = run_episode(task_id, use_llm=use_llm)
            score = result["total_reward"]
            task_scores.append(score)
            all_scores.append(score)
            print(f"  -> Score: {score:.3f} | Diagnosis: {result['diagnosis_made']} (true: {result['true_diagnosis']})")

        results[task_id] = task_scores

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Task':<25} | {'Episodes':>8} | {'Avg Score':>9} | {'Min':>6} | {'Max':>6}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}")
    for task_id in TASKS:
        scores = results[task_id]
        avg = sum(scores) / len(scores)
        mn = min(scores)
        mx = max(scores)
        print(f"{task_id:<25} | {len(scores):>8} | {avg:>9.3f} | {mn:>6.3f} | {mx:>6.3f}")
    print(f"{'='*60}")

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\nOverall average: {overall_avg:.3f}")

    # Exit with error if env seems broken
    if overall_avg == 0:
        print("\nERROR: Average score is 0 — environment may be broken!")
        sys.exit(1)

    print("\nInference complete.")


if __name__ == "__main__":
    main()
