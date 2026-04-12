"""
inference.py — Misinformation Detection Agent
[START] task=<n> env=misinfo model=<model>
[STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import time
import json
import re
import requests
from openai import OpenAI

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client with HF_TOKEN as api_key
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

BASE_URL  = os.getenv("ENV_URL", "http://localhost:7860")
TASKS     = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a misinformation detection expert.
Determine if the news article is real or fake.
Reply ONLY with JSON: {"action_type": "classify", "answer": "fake", "explanation": "reason"}
answer must be exactly real or fake."""

def call_llm(article, task):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task:{task}\nArticle:{article[:600]}"},
            ],
            max_tokens=150,
            temperature=0,
        )
        raw = completion.choices[0].message.content.strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"LLM error: {e}", flush=True)
    return {"action_type": "classify", "answer": "fake", "explanation": "fallback"}

def run_task(task):
    rewards    = []
    error_msg  = "null"
    success    = False
    session_id = None

    print(f"[START] task={task} env=misinfo model={MODEL_NAME}", flush=True)

    try:
        r = requests.post(f"{BASE_URL}/reset", params={"task": task}, timeout=30)
        r.raise_for_status()
        obs        = r.json()
        session_id = obs.get("session_id", "")
        article    = obs.get("article_text", obs.get("text", ""))
        done       = False
        step_num   = 0

        while not done and step_num < 5:
            action_dict = call_llm(article, task)
            try:
                sr = requests.post(
                    f"{BASE_URL}/step",
                    params={"session_id": session_id},
                    json=action_dict,
                    timeout=30,
                )
                sr.raise_for_status()
                obs    = sr.json()
                reward = float(obs.get("score", 0.02))
                done   = bool(obs.get("done", True))
            except Exception as e:
                error_msg = str(e)[:80].replace("\n", " ")
                reward    = 0.02
                done      = True

            step_reward = max(0.02, min(0.97, reward - sum(rewards)))
            rewards.append(step_reward)
            alog = action_dict.get("action_type", "classify")
            if action_dict.get("answer"):
                alog += f":{action_dict['answer']}"

            print(f"[STEP] step={step_num+1} action={alog} reward={step_reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)
            step_num  += 1
            error_msg  = "null"

        success = sum(rewards) >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e)[:80].replace("\n", " ")
        print(f"[STEP] step=1 action=null reward=0.02 done=true error={error_msg}", flush=True)
        rewards = [0.02]

    finally:
        if session_id:
            try:
                requests.delete(f"{BASE_URL}/session", params={"session_id": session_id}, timeout=10)
            except Exception:
                pass

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} score={sum(rewards):.2f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

if __name__ == "__main__":
    time.sleep(2)
    for task in TASKS:
        run_task(task)
        time.sleep(1)
