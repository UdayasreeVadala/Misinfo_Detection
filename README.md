---
title: Misinfo Detection OpenEnv
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---
# Misinformation Detection — OpenEnv Environment

An RL environment where an AI agent investigates news articles and learns to
detect misinformation through multi-step reasoning, source assessment, and
calibrated verdict-giving.

---

## Why this environment?

Misinformation detection is one of the highest-value real-world NLP tasks today.
Existing benchmarks treat it as a **single-shot classification problem** — input
an article, output real/fake. That ignores how humans actually fact-check:
they ask questions, search for corroborating sources, assess publication
credibility, and update their beliefs over multiple steps.

This environment models **the full investigative process**, making it suitable
for training and evaluating agents that reason, not just classify.

---

## Environment description

| Property        | Value                            |
|-----------------|----------------------------------|
| Name            | `misinfo-detection`              |
| Version         | 1.0.0                            |
| Dataset         | GonzaloA/fake_news (HuggingFace) |
| Episode types   | 3 tasks (easy → medium → hard)   |
| Max steps       | 1 / 3 / 5                        |
| Reward range    | 0.0 – 1.0                        |
| Reproducibility | Seeded sample selection, temp=0  |

---

## Tasks

### Easy (1 step)
**Goal:** Classify a news headline as `real` or `fake`.

- Agent receives the article title only
- Single classify action → binary reward (1.0 correct, 0.0 wrong)
- Tests: basic linguistic pattern recognition

**Expected baseline score:** ~0.65

---

### Medium (3 steps)
**Goal:** Investigate a news snippet and deliver a reasoned verdict.

| Step | Action type | What the agent does |
|------|-------------|---------------------|
| 1 | `question` | Ask the most useful clarifying question |
| 2 | `search` | Describe a concrete search strategy |
| 3 | `verdict` | Give real/fake with explanation |

- Rewards partial progress at each step
- Explanation quality scored on logical depth, not just word count

**Expected baseline score:** ~0.45

---

### Hard (5 steps)
**Goal:** Full misinformation investigation with calibrated confidence.

| Step | Action type    | What the agent does                         |
|------|----------------|---------------------------------------------|
| 1 | `question`     | Identify the single most suspicious claim   |
| 2 | `search`       | Design a targeted evidence search           |
| 3 | `assess_source`| Evaluate publication/author credibility     |
| 4 | `cross_check`  | Compare claims against expert consensus     |
| 5 | `verdict`      | Real/fake + confidence (0–1) + full reasoning |

- Rewards source awareness (credibility keywords, publication analysis)
- Calibration bonus: confident-and-correct, or uncertain-and-wrong
- Efficiency bonus for sustained quality across trajectory

**Expected baseline score:** ~0.35

---

## Action space

```json
{
  "action_type": "classify | question | search | assess_source | cross_check | verdict",
  "answer":      "real | fake  (required for classify/verdict)",
  "query":       "free-text   (required for question/search)",
  "explanation": "reasoning   (required for medium/hard steps)",
  "confidence":  0.0–1.0      (used in hard verdict for calibration)"
}
```

## Observation space

```json
{
  "task":         "easy | medium | hard",
  "step":         0,
  "max_steps":    1,
  "article_text": "...",
  "source":       "politics | News | null",
  "prompt":       "What action to take next",
  "score":        0.0,
  "done":         false,
  "feedback":     "Feedback from last action"
}
```

## Reward breakdown

```json
{
  "total":             0.72,
  "correctness":       0.4,
  "reasoning_quality": 0.22,
  "source_awareness":  0.1,
  "efficiency":        0.0,
  "feedback":          "Correct verdict. Reasoning: 0.55. ..."
}
```

---

## Setup & usage

### Local development

```bash
# Clone repo
git clone <your-repo-url>
cd misinfo-env

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal — run baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t misinfo-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... misinfo-env
```

### API usage

```python
import requests

# Start episode
r = requests.post("http://localhost:7860/reset", params={"task": "medium"})
session_id = r.json()["session_id"]
obs = r.json()

# Take a step
r = requests.post(
    "http://localhost:7860/step",
    params={"session_id": session_id},
    json={
        "action_type": "question",
        "query": "Who published this article and what is their track record?",
        "explanation": "Source credibility is the first thing to check."
    }
)
print(r.json())

# Get detailed reward
r = requests.get("http://localhost:7860/reward", params={"session_id": session_id})
print(r.json())
```

### Validate spec compliance

```bash
openenv validate openenv.yaml
```

---

## Baseline scores (gpt-4o-mini, temperature=0)

| Task   | Score | Steps |
|--------|-------|-------|
| easy   | ~0.65 | 1     |
| medium | ~0.45 | 3     |
| hard   | ~0.35 | 5     |

Scores are reproducible: same model + temperature=0 + seeded episode selection
produces consistent results across runs.

---

## Project structure

```
misinfo-env/
├── misinfo_env.py   # Core environment (MisinfoEnv, typed models, graders)
├── server.py        # FastAPI server with session management
├── inference.py     # LLM baseline script (OpenAI API)
├── openenv.yaml     # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## HuggingFace Space

Deployed at: `https://udayasree-misifo-detection.hf.space`

Tagged: `openenv`