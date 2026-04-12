"""
FastAPI Server — Misinformation Detection OpenEnv
===================================================
Endpoints:
  GET  /ping              — health check
  GET  /tasks             — list available tasks with descriptions
  POST /reset             — start a new episode (returns session_id)
  POST /step              — submit action, get reward + next observation
  GET  /state             — current observation for a session
  GET  /reward            — last detailed reward breakdown
Session-based design: each client gets an isolated env via session_id.
Prevents concurrent-request state corruption.
"""

import uuid
from fastapi import FastAPI, HTTPException
from misinfo_env import MisinfoEnv, MisinfoAction, MisinfoObservation, MisinfoReward

app = FastAPI(
    title="Misinformation Detection OpenEnv",
    description="RL environment for training agents to detect misinformation.",
    version="1.0.0",
)

# session_id → MisinfoEnv instance
_sessions: dict[str, MisinfoEnv] = {}

TASK_DESCRIPTIONS = {
    "easy": {
        "name": "easy",
        "steps": 1,
        "description": "Classify a news headline as real or fake in a single step.",
        "difficulty": "easy",
        "max_score": 0.99,
    },
    "medium": {
        "name": "medium",
        "steps": 3,
        "description": (
            "Read a news snippet, ask a clarifying question, devise a search strategy, "
            "then give a final verdict with explanation."
        ),
        "difficulty": "medium",
        "max_score": 0.99,
    },
    "hard": {
        "name": "hard",
        "steps": 5,
        "description": (
            "Full misinformation investigation: identify suspicious claim, search for evidence, "
            "assess source credibility, cross-check against consensus, deliver a calibrated verdict."
        ),
        "difficulty": "hard",
        "max_score": 0.99,
    },
}


def _get_session(session_id: str) -> MisinfoEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Misinfo Detection OpenEnv is running"}

@app.get("/ping")
def ping():
    return {"status": "ok", "env": "misinfo-detection", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list(TASK_DESCRIPTIONS.values())}


@app.post("/reset")
def reset(task: str = "easy"):
    """
    Start a new episode. Returns a session_id that must be passed to /step and /state.
    Each call to /reset creates an isolated session — safe for concurrent agents.
    """
    if task not in TASK_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Choose from: easy, medium, hard.")

    session_id = str(uuid.uuid4())
    env = MisinfoEnv(task=task)
    obs = env.reset()
    _sessions[session_id] = env

    return {"session_id": session_id, **obs.model_dump()}


@app.post("/step")
def step(action: MisinfoAction, session_id: str):
    """
    Submit an action. Returns next observation with cumulative score.
    When done=true the episode is over — call /reset to start a new one.
    """
    env = _get_session(session_id)
    try:
        obs = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs.model_dump()


@app.get("/state")
def state(session_id: str):
    """Return current observation without advancing the episode."""
    env = _get_session(session_id)
    try:
        obs = env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs.model_dump()


@app.get("/reward")
def reward(session_id: str):
    """Return the detailed reward breakdown from the last step."""
    env = _get_session(session_id)
    try:
        r = env.reward()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return r.model_dump()


@app.delete("/session")
def delete_session(session_id: str):
    """Clean up a session when the agent is done."""
    _sessions.pop(session_id, None)
    return {"deleted": session_id}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "misinfo-detection",
        "description": "RL environment for training agents to detect misinformation."
    }

@app.get("/schema")
def schema():
    return {
        "action": MisinfoAction.model_json_schema(),
        "observation": MisinfoObservation.model_json_schema(),
        "state": MisinfoObservation.model_json_schema()
    }

@app.post("/step")
def step(action: MisinfoAction, session_id: str):
    env = _get_session(session_id)
    try:
        obs = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = obs.model_dump()
    result["reward"] = result["score"]  # validators reward field expect chestunnaru!
    return result

@app.post("/mcp")
def mcp(request: dict = {}):
    return {"jsonrpc": "2.0", "id": 1, "result": {"status": "ok"}}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
