"""
Baseline Inference Script — Misinformation Detection Agent
===========================================================
Uses the OpenAI API client (gpt-4o-mini, temperature=0) for reproducible scores.

Required environment variables:
  OPENAI_API_KEY   — OpenAI API key
  ENV_URL          — OpenEnv server URL (default: http://localhost:7860)
  MODEL_NAME       — model to use      (default: gpt-4o-mini)

STDOUT FORMAT (strictly followed by OpenEnv spec):
  [START] task=<name> env=misinfo model=<model>
  [STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...> total=<sum>
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL         = os.getenv("ENV_URL", "http://localhost:7860")
API_KEY          = os.getenv("OPENAI_API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN         = os.getenv("HF_TOKEN")
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASKS            = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

if not API_KEY:
    print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# System prompt — shared across all tasks
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert misinformation analyst trained in media literacy and fact-checking.

Your job is to investigate news articles and determine if they are real or fake.

You will receive instructions about what action to take at each step. Always respond with ONLY a JSON object — no preamble, no markdown fences.

Valid action_type values:
  "classify"      — for easy task, single-step classification
  "question"      — ask a clarifying question (put question in "query")
  "search"        — describe a search strategy (put query in "query")
  "assess_source" — evaluate source credibility (put assessment in "explanation")
  "cross_check"   — compare claims with consensus (put analysis in "explanation")
  "verdict"       — final answer (put "real"/"fake" in "answer", reasoning in "explanation", confidence 0.0-1.0 in "confidence")

Always include "action_type". Include "answer" only for classify/verdict steps.
Include "explanation" for all medium/hard steps.
Respond with ONLY a valid JSON object. No markdown, no backticks, no extra text."""


def build_user_prompt(obs: dict) -> str:
    parts = [
        f"TASK LEVEL: {obs['task']}",
        f"STEP: {obs['step'] + 1} of {obs['max_steps']}",
        "",
        f"ARTICLE:\n{obs['article_text']}",
    ]
    if obs.get("source"):
        parts.append(f"\nSOURCE/SUBJECT: {obs['source']}")
    parts += [
        "",
        f"INSTRUCTION: {obs['prompt']}",
        "",
        "Respond with ONLY a JSON object.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call — temperature=0 for reproducibility
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Run one full episode for a given task
# ---------------------------------------------------------------------------

def run_task(task: str) -> float:
    rewards    = []
    error_msg  = "null"
    success    = False
    session_id = None

    print(f"[START] task={task} env=misinfo model={MODEL_NAME}", flush=True)

    try:
        # 1. Reset — get session_id + initial observation
        reset_resp = requests.post(f"{BASE_URL}/reset", params={"task": task})
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        session_id = obs["session_id"]

        # 2. Run steps until done
        step_num = 0
        while not obs.get("done", False):
            user_prompt = build_user_prompt(obs)

            try:
                action_dict = call_llm(SYSTEM_PROMPT, user_prompt)
            except (json.JSONDecodeError, Exception) as e:
                # Fallback: emit a safe action so episode can continue
                is_last = obs["step"] >= obs["max_steps"] - 1
                action_dict = {
                    "action_type": "verdict" if is_last else "classify",
                    "answer":      "fake",
                    "explanation": "Unable to parse LLM response, defaulting to fake.",
                }
                error_msg = f"LLM parse error: {e}"

            step_resp = requests.post(
                f"{BASE_URL}/step",
                params={"session_id": session_id},
                json=action_dict,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            reward      = float(obs.get("score", 0.0))
            step_reward = reward - sum(rewards)   # score is cumulative → extract delta
            rewards.append(step_reward)

            action_log = action_dict.get("action_type", "unknown")
            if action_dict.get("answer"):
                action_log += f":{action_dict['answer']}"

            print(
                f"[STEP] step={step_num + 1} "
                f"action={action_log} "
                f"reward={step_reward:.2f} "
                f"done={str(obs.get('done', False)).lower()} "
                f"error={error_msg}",
                flush=True,
            )
            step_num  += 1
            error_msg  = "null"

        total_reward = sum(rewards)
        success      = total_reward >= SUCCESS_THRESHOLD

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"[STEP] step=1 action=null reward=0.00 done=true error={error_msg}", flush=True)
        rewards = [0.0]

    finally:
        # Clean up session
        if session_id:
            try:
                requests.delete(f"{BASE_URL}/session", params={"session_id": session_id})
            except Exception:
                pass

    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    total_reward = sum(rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} "
        f"rewards={rewards_str} "
        f"total={total_reward:.2f}",
        flush=True,
    )
    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running baseline agent: {MODEL_NAME} on {BASE_URL}", flush=True)
    print("=" * 60, flush=True)
    time.sleep(2)  # give server time to be ready

    totals = {}
    for task in TASKS:
        score = run_task(task)
        totals[task] = score
        time.sleep(1)

    print("=" * 60, flush=True)
    print("BASELINE SUMMARY", flush=True)
    for task, score in totals.items():
        print(f"  {task:8s}: {score:.2f}", flush=True)
    print(f"  {'AVERAGE':8s}: {sum(totals.values()) / len(totals):.2f}", flush=True)