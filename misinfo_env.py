```python
"""
Misinformation Detection Environment (FINAL - VALIDATOR SAFE)
"""

import random
from pydantic import BaseModel
from typing import Optional, Literal

# ======================
# SAFE SCORE CONSTANTS
# ======================
EPSILON_SCORE = 0.01
MAX_SCORE = 0.99


def _strict_score(value: float) -> float:
    try:
        value = float(value)
    except:
        value = 0.5

    if value <= 0:
        return EPSILON_SCORE
    if value >= 1:
        return MAX_SCORE

    return round(value, 2)


# ======================
# DATA MODELS
# ======================

class MisinfoAction(BaseModel):
    answer: Optional[Literal["real", "fake"]] = None
    explanation: Optional[str] = None
    query: Optional[str] = None
    confidence: Optional[float] = None


class MisinfoReward(BaseModel):
    total: float
    correctness: float
    reasoning_quality: float
    source_awareness: float
    efficiency: float
    feedback: str


# ======================
# HELPER FUNCTIONS
# ======================

def _empty_reward():
    return MisinfoReward(
        total=EPSILON_SCORE,
        correctness=EPSILON_SCORE,
        reasoning_quality=EPSILON_SCORE,
        source_awareness=EPSILON_SCORE,
        efficiency=EPSILON_SCORE,
        feedback=""
    )


def _finalize_reward(r: MisinfoReward):
    if r.total is None:
        r.total = 0.5

    r.total = _strict_score(r.total)
    r.correctness = _strict_score(r.correctness)
    r.reasoning_quality = _strict_score(r.reasoning_quality)
    r.source_awareness = _strict_score(r.source_awareness)
    r.efficiency = _strict_score(r.efficiency)

    return r


def _reasoning_score(explanation: str, min_words: int):
    if not explanation:
        return EPSILON_SCORE

    words = explanation.split()
    score = 0.0

    if len(words) >= min_words:
        score += 0.6
    elif len(words) >= min_words // 2:
        score += 0.3

    if any(k in explanation.lower() for k in ["because", "evidence", "suggests", "likely"]):
        score += 0.3

    return _strict_score(score)


# ======================
# SAMPLE DATA
# ======================

SAMPLE_DATA = [
    {"text": "Government confirms new education policy update", "label": "real"},
    {"text": "Aliens found living under the ocean", "label": "fake"},
    {"text": "Stock market hits record high this week", "label": "real"},
    {"text": "Drinking 10 liters of water cures all diseases", "label": "fake"},
]


# ======================
# ENVIRONMENT CLASS
# ======================

class MisinfoEnv:

    def __init__(self):
        self.sample = None
        self.done = False
        self.last_reward = None

    def reset(self):
        self.sample = random.choice(SAMPLE_DATA)
        self.done = False
        self.last_reward = None

        return {
            "text": self.sample["text"],
            "task": "easy",
            "done": False,
            "feedback": "Classify as real or fake"
        }

    def step(self, action: dict):
        action = MisinfoAction(**action)

        correct = (action.answer or "").lower() == self.sample["label"]

        correctness = 0.9 if correct else 0.1
        reasoning = _reasoning_score(action.explanation or "", 5)

        total = correctness * 0.7 + reasoning * 0.3

        reward = MisinfoReward(
            total=total,
            correctness=correctness,
            reasoning_quality=reasoning,
            source_awareness=0.1,
            efficiency=0.1,
            feedback="Correct!" if correct else f"Wrong! It was {self.sample['label']}"
        )

        reward = _finalize_reward(reward)

        self.done = True
        self.last_reward = reward

        return {
            "score": reward.total,
            "done": True,
            "feedback": reward.feedback
        }

    def state(self):
        return {
            "text": self.sample["text"],
            "done": self.done
        }
```
