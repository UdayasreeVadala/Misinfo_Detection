"""
inference.py — Misinformation Detection Agent
===============================================
[START] task=<name> env=misinfo model=<model>
[STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import os
import time
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")    or os.getenv("OPENAI_API_KEY", "dummy-key")
BASE_URL     = os.getenv("ENV_URL",      "http://localhost:7860")
TASKS        = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a misinformation detection expert.
Given a news article, determine if it is real or fake.
For easy tasks: reply with JSON {"action_type": "classify", "answer": "real" or "fake"}
For medium tasks: reply with JSON {"action_type": "verdict", "answer": "real" or "fake", "explanation": "your reasoning"}
For hard tasks: reply with JSON {"action_type": "verdict", "answer": "real" or "fake", "explanation": "detailed reasoning", "confidence": 0.0-1.0}
Reply ONLY with the JSON object, nothing else."""

# ── Run One Task ─────────────────────────────────────────────────────────────
def run_task(task: str):
    rewards   = []
    error_msg = "null"
    success   = False

    print(f"[START] task={task} env=misinfo model={MODEL_NAME}", flush=True)

    try:
        # Reset
        reset_resp = requests.post(f"{BASE_URL}/reset", params={"task": task})
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        session_id   = obs["session_id"]
        article_text = obs.get("article_text", "")

        step_num = 0
        done = False

        while not done and step_num < 5:
            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Task level: {task}\n\nArticle:\n{article_text[:1000]}"},
                    ],
                    max_tokens=200,
                    temperature=0,
                )
                raw = completion.choices[0].message.content.strip()

                import json, re
                try:
                    match = re.search(r"\{.*\}", raw, re.DOTALL)
                    action_dict = json.loads(match.group()) if match else {}
                except Exception:
                    action_dict = {}

                if not action_dict.get("action_type"):
                    action_dict = {"action_type": "classify", "answer": "fake", "explanation": "fallback"}

            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                action_dict = {"action_type": "classify", "answer": "fake", "explanation": "llm error"}

            # Step
            step_resp = requests.post(
                f"{BASE_URL}/step",
                params={"session_id": session_id},
                json=action_dict,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            reward = float(obs.get("score", 0.0))
            done   = bool(obs.get("done", True))
            step_reward = reward - sum(rewards)
            rewards.append(max(step_reward, 0.0))

            action_log = action_dict.get("action_type", "classify")
            if action_dict.get("answer"):
                action_log += f":{action_dict['answer']}"

            print(
                f"[STEP] step={step_num+1} action={action_log} "
                f"reward={rewards[-1]:.2f} done={str(done).lower()} error={error_msg}",
                flush=True,
            )
            step_num += 1
            error_msg = "null"

        success = sum(rewards) >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={error_msg}", flush=True)
        rewards = [0.0]

    finally:
        try:
            requests.delete(f"{BASE_URL}/session", params={"session_id": session_id})
        except Exception:
            pass

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} rewards={rewards_str}",
        flush=True,
    )

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    time.sleep(2)
    for task in TASKS:
        run_task(task)
        time.sleep(1)
