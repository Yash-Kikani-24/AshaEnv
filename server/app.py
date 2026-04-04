from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from asha_env.env import AshaEnv

app = FastAPI(
    title="ASHA Agent Environment",
    description="RL environment simulating ASHA worker patient consultations in rural India",
    version="1.0.0",
)

# Global environment instance
env = AshaEnv()


# --- Pydantic Models ---

class ResetRequest(BaseModel):
    task_id: Optional[str] = "medium_consultation"


class StepRequest(BaseModel):
    action: str
    episode_id: str


class ResetResponse(BaseModel):
    patient: dict
    asha_context: dict
    available_actions: list[str]
    episode_id: str


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    episode_id: Optional[str]
    true_diagnosis: Optional[str]
    comorbidities: list
    true_symptoms: list
    history: list
    trajectory: list
    step_count: int
    total_reward: float
    done: bool
    diagnosis_made: Optional[str]
    referral_made: Optional[str]
    treatments_given: list
    patient: Optional[dict]


class HealthResponse(BaseModel):
    status: str


# --- Endpoints ---

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    try:
        observation = env.reset(task_id=request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return observation


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    # Validate episode ID
    if env.episode_id is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    if request.episode_id != env.episode_id:
        raise HTTPException(
            status_code=400,
            detail=f"Episode ID mismatch. Expected: {env.episode_id}, got: {request.episode_id}",
        )

    observation, reward, done, info = env.step(request.action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=StateResponse)
def get_state():
    if env.episode_id is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return env.get_state()


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}
