"""
inference.py — LLM agent that plays episodes against the ASHA environment over HTTP.

This is the main entry point called by the OpenEnv evaluation platform. It connects
to the running environment server (server/app.py), runs an LLM through full episodes
on all three task difficulties, and prints a results summary.

HOW IT WORKS:
    1. Calls POST /reset to start a new patient episode and get the initial observation.
    2. Feeds the observation to an LLM via an OpenAI-compatible chat API.
    3. Parses the LLM's response into a valid action string.
    4. Calls POST /step with that action, gets updated observation + reward.
    5. Repeats until done=True, then calls GET /state to retrieve the final outcome.
    6. Prints a per-task and overall results table.

FALLBACK MODE:
    If HF_TOKEN or MODEL_NAME are not set (or MODEL_NAME="test"), the script runs
    without an LLM — it just picks the first available action each step. This is
    useful for smoke-testing the server connection without needing API credentials.

REQUIRED ENVIRONMENT VARIABLES:
    API_BASE_URL  — URL of the environment server (default: http://localhost:7860)
    HF_TOKEN      — API key for the LLM provider (HuggingFace or OpenAI-compatible)
    MODEL_NAME    — model identifier, e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    LLM_API_URL   — (optional) LLM endpoint URL (default: OpenAI chat completions)

HOW TO RUN LOCALLY:
    # Start the server first:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Then in a separate terminal:
    API_BASE_URL=http://localhost:7860 HF_TOKEN=<key> MODEL_NAME=<model> python inference.py
"""

import os
import re
import sys
import json
import httpx

# ---------------------------------------------------------------------------
# Configuration — all values can be overridden via environment variables.
# ---------------------------------------------------------------------------

# Base URL of the running ASHA environment server.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")

# API key for the LLM. Checks HF_TOKEN first, then API_KEY as a fallback alias.
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("API_KEY", ""))

# Which LLM model to use. Default is Llama-3 8B via HuggingFace Inference API.
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# System prompt that sets up the LLM's role as an ASHA worker.
# Key constraints communicated to the LLM:
#   - It must output EXACTLY ONE action string from available_actions.
#   - It should prioritise high-value symptoms and escalate emergencies fast.
SYSTEM_PROMPT = """You are an ASHA (Accredited Social Health Activist) worker in rural India.
You are conducting a consultation for a pregnant woman or a newborn baby. Your job is to:
1. Ask relevant symptom questions to identify maternal or neonatal danger signs
2. Order appropriate tests from your ASHA kit when needed
3. Make a diagnosis when you have enough information
4. Treat with available medicines or refer to the appropriate facility

You have a limited medical kit with: IFA tablets, calcium tablets, ORS, zinc, paracetamol, misoprostol.
You can run tests: BP monitor, thermometer, haemoglobin strip, urine dipstick, weighing scale.

IMPORTANT: You must respond with EXACTLY ONE action from the available_actions list.
Respond with just the action string, nothing else. Example: ask_symptom:pallor

Be efficient — ask the most informative questions first. Don't waste steps on unlikely symptoms.
If you suspect an emergency (e.g. eclampsia, birth asphyxia, haemorrhage), refer immediately."""

# Tasks to evaluate — matches the three difficulty levels in AshaEnv.
TASKS = ["easy_diagnosis", "medium_consultation", "hard_complex_case"]

# Number of episodes to run per task. 5 gives a reasonable average without
# taking too long during platform evaluation.
EPISODES_PER_TASK = 5


def parse_action(raw_text: str, available_actions: list[str]) -> str:
    """
    Extract a valid action string from the LLM's raw text output.

    LLMs don't always follow the "just the action string" instruction perfectly —
    they may add explanation, punctuation, or casing differences. This function
    tries progressively looser matching strategies until it finds something valid.

    Matching order (first match wins):
        1. Exact match against available_actions.
        2. Regex extraction of any "action_type:value" pattern in the text,
           checked against available_actions.
        3. Case-insensitive exact match.
        4. Substring match — any available action that appears inside the text.
        5. Fallback — first available action (ensures the episode never gets stuck).

    Args:
        raw_text:         The raw string returned by the LLM.
        available_actions: List of valid action strings from the current observation.

    Returns:
        A valid action string from available_actions, or the raw text if the list
        is empty (shouldn't happen in normal operation).
    """
    raw = raw_text.strip()

    # --- Strategy 1: Exact match ---
    if raw in available_actions:
        return raw

    # --- Strategy 2: Regex extraction ---
    # Search for any "action_type:value" pattern within the LLM's response.
    # Useful when the LLM says something like "I would choose ask_symptom:pallor because..."
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

    # --- Strategy 3: Case-insensitive match ---
    raw_lower = raw.lower().strip()
    for action in available_actions:
        if action.lower() == raw_lower:
            return action

    # --- Strategy 4: Substring match ---
    # Catches cases where the LLM wraps the action in quotes or a sentence.
    for action in available_actions:
        if action in raw:
            return action

    # --- Strategy 5: Fallback ---
    # If nothing matched, pick the first valid action so the episode can continue.
    if available_actions:
        return available_actions[0]

    return raw


def call_llm(messages: list[dict]) -> str:
    """
    Send the conversation history to the LLM and return its reply text.

    Uses an OpenAI-compatible chat completions endpoint. The LLM_API_URL environment
    variable can point this at HuggingFace Inference API, vLLM, or any other
    OpenAI-compatible server.

    Settings:
        max_tokens=100  — actions are short strings; no need for longer responses.
        temperature=0.3 — low temperature for more deterministic action selection.

    Args:
        messages: Full conversation history in OpenAI message format
                  (list of {"role": "system"|"user"|"assistant", "content": str}).

    Returns:
        The LLM's reply as a plain string, or "" if the API call failed.
    """
    try:
        # Default to OpenAI's endpoint; override with LLM_API_URL for other providers.
        api_url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 100,   # Actions are short — no need for more tokens.
            "temperature": 0.3,  # Low temperature = more consistent action selection.
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""  # Return empty string; parse_action will fall back to first available action.


def run_episode(task_id: str, use_llm: bool = True) -> dict:
    """
    Run one complete episode against the environment server and return results.

    Opens an HTTP connection to the server, resets the environment to start a
    fresh patient, then loops until done=True — each iteration consulting the LLM
    (or fallback) for the next action. After the episode ends, fetches the full
    ground-truth state for the results summary.

    The full conversation history (system prompt + all user/assistant turns) is
    maintained across steps so the LLM has context about what has already happened.

    Args:
        task_id:  One of TASKS — controls difficulty and allowed diseases.
        use_llm:  If True, uses the configured LLM. If False, picks the first
                  available action each step (smoke-test / fallback mode).

    Returns:
        Dict with: task_id, episode_id, steps taken, total_reward, true_diagnosis,
        diagnosis_made, referral_made.
    """
    with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
        # --- Start a new episode ---
        resp = client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
        episode_id = obs["episode_id"]  # Must be sent with every /step call.

        # Conversation history — system prompt is pre-loaded, user/assistant turns
        # are appended each step so the LLM remembers what it already asked.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            available = obs.get("available_actions", [])
            if not available:
                break

            if use_llm:
                # Serialise the full observation as JSON so the LLM sees patient
                # info, known symptoms, ASHA context, and the list of valid actions.
                user_msg = (
                    f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
                    f"Choose ONE action from available_actions. Respond with just the action string."
                )
                messages.append({"role": "user", "content": user_msg})
                raw_response = call_llm(messages)
                action = parse_action(raw_response, available)
                # Append the chosen action as the assistant turn so the LLM
                # knows what it did last time in future steps.
                messages.append({"role": "assistant", "content": action})
            else:
                # Fallback mode — no LLM, just pick the first action mechanically.
                action = available[0]

            # --- Send action to environment ---
            resp = client.post("/step", json={"action": action, "episode_id": episode_id})
            resp.raise_for_status()
            step_data = resp.json()

            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward
            steps += 1

            print(f"    Step {steps}: {action} -> reward={reward:.3f}")

        # --- Fetch ground-truth after episode ends ---
        # /state exposes hidden fields (true_diagnosis, true_symptoms) used
        # to show whether the LLM diagnosed correctly.
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
    """
    Run the full evaluation benchmark and print a summary table.

    Detects whether LLM credentials are available and chooses LLM or fallback
    mode accordingly. Runs EPISODES_PER_TASK episodes for each task, collects
    scores, and prints per-task and overall statistics.

    Exits with code 1 if the overall average score is exactly 0 — this most
    likely means the environment server is not reachable or is broken.
    """
    # LLM mode requires both a token and a non-test model name.
    use_llm = HF_TOKEN and MODEL_NAME and MODEL_NAME != "test"
    if not use_llm:
        print("No LLM configured (HF_TOKEN/MODEL_NAME missing or MODEL_NAME=test).")
        print("Running in fallback mode (first available action).\n")

    results = {}    # results[task_id] = list of total_reward floats
    all_scores = [] # Flat list of all scores for the overall average

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

    # --- Print summary table ---
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

    # A score of exactly 0 across all episodes most likely means the environment
    # server is unreachable or all requests failed silently.
    if overall_avg == 0:
        print("\nERROR: Average score is 0 — environment may be broken!")
        sys.exit(1)

    print("\nInference complete.")


if __name__ == "__main__":
    main()
