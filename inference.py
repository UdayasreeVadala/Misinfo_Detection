"""
inference.py — Misinformation Detection Agent
[START] task=<n> env=misinfo model=<model>
[STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import time
import json
import re
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy-key"
BASE_URL     = os.getenv("ENV_URL",  "http://localhost:7860")
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

try:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    client = None

SYSTEM_PROMPT = """You are a misinformation detection expert.
Given a news article, determine if it is real or fake.
Reply ONLY with a JSON object like:
{"action_type": "classify", "answer": "real", "explanation": "reason here"}
answer must be exactly "real" or "fake"."""

def call_llm(article: str, task: str) -> dict:
    if client is None:
        return {"action_type": "classify", "answer": "fake", "explanation": "no client"}
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task}\n\nArticle:\n{article[:800]}"},
            ],
            max_tokens=150,
            temperature=0,
        )
        raw = completion.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"action_type": "classify", "answer": "fake", "explanation": "fallback"}

def run_task(task: str):
    rewards   = []
    error_msg = "null"
    success   = False
    session_id = None

    print(f"[START] task={task} env=misinfo model={MODEL_NAME}", flush=True)

    try:
        reset_resp = requests.post(f"{BASE_URL}/reset", params={"task": task}, timeout=30)
        reset_resp.raise_for_status()
        obs        = reset_resp.json()
        session_id = obs.get("session_id", "")
        article    = obs.get("article_text", obs.get("text", ""))

        step_num = 0
        done     = False

        while not done and step_num < 5:
            action_dict = call_llm(article, task)

            try:
                step_resp = requests.post(
                    f"{BASE_URL}/step",
                    params={"session_id": session_id},
                    json=action_dict,
                    timeout=30,
                )
                step_resp.raise_for_status()
                obs    = step_resp.json()
                reward = float(obs.get("score", 0.0))
                done   = bool(obs.get("done", True))
            except Exception as e:
                error_msg = str(e).replace("\n", " ")[:100]
                reward = 0.0
                done   = True

            step_reward = max(reward - sum(rewards), 0.0)
            rewards.append(step_reward)

            action_log = action_dict.get("action_type", "classify")
            if action_dict.get("answer"):
                action_log += f":{action_dict['answer']}"

            print(
                f"[STEP] step={step_num+1} action={action_log} "
                f"reward={step_reward:.2f} done={str(done).lower()} error={error_msg}",
                flush=True,
            )
            step_num  += 1
            error_msg  = "null"

        success = sum(rewards) >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:100]
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={error_msg}", flush=True)
        rewards = [0.0]

    finally:
        if session_id:
            try:
                requests.delete(f"{BASE_URL}/session", params={"session_id": session_id}, timeout=10)
            except Exception:
                pass

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    time.sleep(2)
    for task in TASKS:
        run_task(task)
        time.sleep(1)
