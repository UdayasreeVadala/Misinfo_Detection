"""
Baseline Inference Script — Misinformation Detection Agent
===========================================================
Uses the OpenAI API client (gpt-4o-mini, temperature=0) for reproducible scores.

Required environment variables:
  OPENAI_API_KEY   — OpenAI API key (falls back to HF_TOKEN or META_API_KEY)
  API_BASE_URL     — LLM API base URL (default: https://api.openai.com/v1)
  MODEL_NAME       — model to use (default: gpt-4o-mini)
  HF_TOKEN         — Hugging Face token (used as fallback API key)
  ENV_URL          — OpenEnv server URL (default: http://localhost:7860)

STDOUT FORMAT:
  [START] task=<name> env=misinfo model=<model>
  [STEP]  step=<n> action=<text> reward=<0.00> done=<bool> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...> total=<sum>
"""

import os
import sys
import json
import time
import requests
from typing import Tuple

BASE_URL          = os.getenv("ENV_URL", "http://localhost:7860")
API_KEY           = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("META_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
)
API_BASE_URL      = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
MODEL_NAME        = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASKS             = ["easy", "medium", "hard"]
SUCCESS_THRESHOLD = 0.5

LLM_AVAILABLE = bool(API_KEY)

if not LLM_AVAILABLE:
    print(
        "WARNING: No API key found (OPENAI_API_KEY / META_API_KEY / HF_TOKEN). "
        "Running in heuristic-fallback mode.",
        file=sys.stderr,
        flush=True,
    )

client = None
if LLM_AVAILABLE:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"WARNING: Could not initialise OpenAI client: {e}", file=sys.stderr, flush=True)
        LLM_AVAILABLE = False

FAKE_KEYWORDS = {
    "shocking", "unbelievable", "secret", "they don't want you",
    "exposed", "hoax", "conspiracy", "miracle", "cure", "banned",
    "government hides", "mainstream media", "what they aren't telling",
    "wake up", "sheeple", "deep state", "illuminati", "chemtrail",
    "crisis actor", "false flag", "plandemic", "microchip",
}

REAL_KEYWORDS = {
    "according to", "researchers", "study", "published", "reported",
    "officials", "confirmed", "percent", "statistics", "data",
    "university", "hospital", "government", "announced", "survey",
}


def heuristic_verdict(text: str) -> str:
    lower = text.lower()
    fake_hits = sum(1 for kw in FAKE_KEYWORDS if kw in lower)
    real_hits = sum(1 for kw in REAL_KEYWORDS if kw in lower)
    return "fake" if fake_hits > real_hits else "real"


SYSTEM_PROMPT = """You are an expert misinformation analyst trained in media literacy and fact-checking.

Your job is to investigate news articles and determine if they are real or fake.

Always respond with ONLY a JSON object — no preamble, no markdown fences.

Valid action_type values:
  "classify"      — easy task single-step classification
  "question"      — ask a clarifying question (put question in "query")
  "search"        — describe a search strategy (put query in "query")
  "assess_source" — evaluate source credibility (put assessment in "explanation")
  "cross_check"   — compare claims with consensus (put analysis in "explanation")
  "verdict"       — final answer ("real"/"fake" in "answer", reasoning in "explanation", confidence 0.0-1.0 in "confidence")

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

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def fallback_action(obs: dict) -> dict:
    task      = obs.get("task", "easy")
    step      = obs.get("step", 0)
    max_steps = obs.get("max_steps", 1)
    text      = obs.get("article_text", "")
    verdict   = heuristic_verdict(text)

    if task == "easy":
        return {"action_type": "classify", "answer": verdict}

    if task == "medium":
        if step == 0:
            return {
                "action_type": "question",
                "query": "What is the original source of this article and when was it published?",
            }
        if step == 1:
            return {
                "action_type": "search",
                "query": f"fact-check: {text[:80]}",
            }
        return {
            "action_type": "verdict",
            "answer": verdict,
            "explanation": (
                "Based on linguistic patterns and source assessment, "
                f"this article appears to be {verdict}."
            ),
            "confidence": 0.65,
        }

    if task == "hard":
        if step == 0:
            return {
                "action_type": "question",
                "query": "What specific claim in this article requires fact-checking?",
            }
        if step == 1:
            return {
                "action_type": "search",
                "query": f"verify claim: {text[:80]}",
            }
        if step == 2:
            return {
                "action_type": "assess_source",
                "explanation": (
                    "Assessing source credibility based on language, citation style, "
                    "and presence of verifiable facts. "
                    "Sensationalist language and lack of citations reduce credibility."
                ),
            }
        if step == 3:
            return {
                "action_type": "cross_check",
                "explanation": (
                    "Cross-checking claims against known facts and consensus reporting. "
                    "Comparing with established news sources and scientific literature."
                ),
            }
        return {
            "action_type": "verdict",
            "answer": verdict,
            "explanation": (
                "After multi-step analysis including source assessment and cross-checking, "
                f"this article is assessed as {verdict} based on linguistic markers, "
                "source credibility, and consistency with known facts."
            ),
            "confidence": 0.70,
        }

    return {"action_type": "classify", "answer": verdict}


def get_action(obs: dict) -> Tuple[dict, str]:
    if LLM_AVAILABLE and client is not None:
        user_prompt = build_user_prompt(obs)
        try:
            action = call_llm(SYSTEM_PROMPT, user_prompt)
            return action, "null"
        except Exception as e:
            err = str(e).replace("\n", " ")[:80]
            return fallback_action(obs), f"LLM_error:{err}"
    return fallback_action(obs), "null"


def run_task(task: str) -> float:
    rewards    = []
    error_msg  = "null"
    success    = False
    session_id = None
    step_num   = 0

    print(f"[START] task={task} env=misinfo model={MODEL_NAME}", flush=True)

    try:
        reset_resp = requests.post(f"{BASE_URL}/reset", params={"task": task}, timeout=30)
        reset_resp.raise_for_status()
        obs = reset_resp.json()
        session_id = obs["session_id"]

        while not obs.get("done", False):
            action_dict, error_msg = get_action(obs)

            step_resp = requests.post(
                f"{BASE_URL}/step",
                params={"session_id": session_id},
                json=action_dict,
                timeout=30,
            )
            step_resp.raise_for_status()
            obs = step_resp.json()

            reward = max(0.01, min(0.99, float(obs.get("score", 0.0))))
            step_reward = max(0.01, round(reward - sum(rewards), 4))
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
        error_msg = str(e).replace("\n", " ")[:100]
        print(
            f"[STEP] step={step_num + 1} "
            f"action=null "
            f"reward=0.00 "
            f"done=true "
            f"error={error_msg}",
            flush=True,
        )
        if not rewards:
            rewards = [0.01]

    finally:
        if session_id:
            try:
                requests.delete(
                    f"{BASE_URL}/session",
                    params={"session_id": session_id},
                    timeout=10,
                )
            except Exception:
                pass

    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    total_reward = max(0.01, min(0.99, sum(rewards)))
    success      = total_reward >= SUCCESS_THRESHOLD

    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} "
        f"rewards={rewards_str} "
        f"total={total_reward:.2f}",
        flush=True,
    )
    return total_reward


if __name__ == "__main__":
    mode = "LLM" if LLM_AVAILABLE else "heuristic-fallback"
    print(f"Running baseline agent: {MODEL_NAME} on {BASE_URL} [{mode}]", flush=True)
    print("=" * 60, flush=True)
    time.sleep(2)

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
