"""
inference.py — Misinformation Detection Agent
[START] task=<n> env=misinfo model=<model>
[STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import sys
import time
import json
import re
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "dummy-key"
BASE_URL     = os.environ.get("ENV_URL", "http://localhost:7860")
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

# Safe OpenAI client init
client = None
try:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    pass

SYSTEM_PROMPT = """You are a misinformation detection expert.
Determine if the news article is real or fake.
Reply ONLY with JSON: {"action_type": "classify", "answer": "fake", "explanation": "reason"}
answer must be exactly real or fake."""

def call_llm(article: str, task: str) -> dict:
    if client:
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
        except Exception:
            pass
    return {"action_type": "classify", "answer": "fake", "explanation": "fallback"}

def run_task(task: str):
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
                reward = float(obs.get("score", 0.0))
                done   = bool(obs.get("done", True))
            except Exception as e:
                error_msg = str(e)[:80].replace("\n", " ")
                reward    = 0.0
                done      = True

            step_reward = max(reward - sum(rewards), 0.0)
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
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={error_msg}", flush=True)
        rewards = [0.0]

    finally:
        if session_id:
            try:
                requests.delete(f"{BASE_URL}/session", params={"session_id": session_id}, timeout=10)
            except Exception:
                pass

    print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

if __name__ == "__main__":
    time.sleep(2)
    for task in TASKS:
        run_task(task)
        time.sleep(1)
