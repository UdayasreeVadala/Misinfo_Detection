"""
Misinformation Detection Environment
======================================
OpenEnv-compatible RL environment where an AI agent learns
to detect and reason about misinformation across 3 difficulty levels.

Multi-step episode flow (per task):
  easy   — 1 step  : classify the headline (real/fake)
  medium — 3 steps : observe snippet → ask a clarifying question → give final verdict
  hard   — 5 steps : observe article → search for evidence → assess source → cross-check → final verdict

Reward is partial and trajectory-level — agent earns signal at every step.
"""

import random
import hashlib
from datasets import load_dataset
from pydantic import BaseModel
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# Typed Models  (OpenEnv spec requires all three)
# ---------------------------------------------------------------------------

class MisinfoAction(BaseModel):
    action_type: Literal["classify", "question", "search", "assess_source", "cross_check", "verdict"]
    answer: Optional[str] = None          # "real" | "fake"  (for classify / verdict steps)
    query: Optional[str] = None           # free-text for question / search steps
    explanation: Optional[str] = None     # reasoning — required from medium onwards
    confidence: Optional[float] = None    # 0.0–1.0, used in hard grader calibration


class MisinfoObservation(BaseModel):
    task: str
    step: int
    max_steps: int
    article_text: str
    source: Optional[str] = None
    prompt: str                           # what the agent should do next
    score: float
    done: bool
    feedback: str


class MisinfoReward(BaseModel):
    total: float
    correctness: float
    reasoning_quality: float
    source_awareness: float
    efficiency: float                     # bonus for solving in fewer steps
    feedback: str


# ---------------------------------------------------------------------------
# Dataset  (loaded once at import time)
# ---------------------------------------------------------------------------

print("Loading dataset...")
_dataset = load_dataset("GonzaloA/fake_news", split="train")

_real = [x for x in _dataset if x["label"] == 1]
_fake = [x for x in _dataset if x["label"] == 0]

random.shuffle(_real)
random.shuffle(_fake)
REAL_POOL = _real[:500]
FAKE_POOL = _fake[:500]
ALL_POOL  = REAL_POOL + FAKE_POOL

print(f"Dataset ready — real: {len(REAL_POOL)}, fake: {len(FAKE_POOL)}")


# ---------------------------------------------------------------------------
# Reproducible sample selection
# ---------------------------------------------------------------------------

def _get_sample(task: str, episode: int) -> dict:
    """
    Deterministic sample selection: same task + episode index → same article.
    Uses a hash so the episode seed doesn't just walk the list in order.
    """
    seed_str = f"{task}-{episode}"
    idx = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % len(ALL_POOL)
    sample = ALL_POOL[idx]
    label  = "real" if sample["label"] == 1 else "fake"

    full_text = sample.get("text", "") or ""

    if task == "easy":
        text = sample["title"] or full_text[:120]
    elif task == "medium":
        text = full_text[:400] if full_text else sample["title"]
    else:
        text = full_text[:800] if full_text else sample["title"]

    return {
        "title":  sample.get("title", ""),
        "text":   text,
        "label":  label,
        "source": sample.get("subject", "unknown"),
        "full":   full_text,
    }


# ---------------------------------------------------------------------------
# Step configs per difficulty
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "easy": {
        "max_steps": 1,
        "step_prompts": [
            "Read the headline and classify it as 'real' or 'fake'. Use action_type='classify'."
        ],
    },
    "medium": {
        "max_steps": 3,
        "step_prompts": [
            "Read the article snippet. Use action_type='question' to ask one clarifying question about what would help you verify this claim.",
            "Based on your question, use action_type='search' to describe what evidence you would look for online.",
            "Now give your final verdict. Use action_type='verdict' with answer='real' or 'fake' and a clear explanation.",
        ],
    },
    "hard": {
        "max_steps": 5,
        "step_prompts": [
            "Read the full article carefully. Use action_type='question' to identify the single most suspicious claim.",
            "Use action_type='search' to describe the specific search query and sources you would check to verify that claim.",
            "Use action_type='assess_source' to evaluate the credibility of the article's source/publication.",
            "Use action_type='cross_check' to compare this article's claims with what credible sources would say.",
            "Give your final verdict. Use action_type='verdict' with answer='real'/'fake', a confidence score 0.0–1.0, and a detailed explanation citing your earlier steps.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _reasoning_score(explanation: str, min_words: int, keywords: list[str] = None) -> float:
    """
    Score explanation quality:
      - 0.5 for meeting minimum word threshold
      - 0.3 for logical structure markers (because, therefore, however, suggests, indicates, etc.)
      - 0.2 for domain keywords if provided
    """
    if not explanation:
        return 0.0
    words = explanation.lower().split()
    score = 0.0

    # Word count
    if len(words) >= min_words:
        score += 0.5
    elif len(words) >= min_words // 2:
        score += 0.25

    # Logical connectors
    logic_markers = ["because", "therefore", "however", "suggests", "indicates",
                     "evidence", "appears", "likely", "unlikely", "based on",
                     "this implies", "which means", "consistent with", "inconsistent"]
    if any(m in explanation.lower() for m in logic_markers):
        score += 0.3

    # Domain keywords
    if keywords:
        if any(kw in explanation.lower() for kw in keywords):
            score += 0.2

    return min(score, 1.0)


def grade_step_easy(step: int, action: MisinfoAction, sample: dict, history: list) -> MisinfoReward:
    correct = (action.answer or "").lower().strip() == sample["label"]
    correctness = 1.0 if correct else 0.0

    return MisinfoReward(
        total=correctness,
        correctness=correctness,
        reasoning_quality=0.0,
        source_awareness=0.0,
        efficiency=0.0,
        feedback="Correct!" if correct else f"Wrong — it was {sample['label']}.",
    )


def grade_step_medium(step: int, action: MisinfoAction, sample: dict, history: list) -> MisinfoReward:
    r = MisinfoReward(total=0.0, correctness=0.0, reasoning_quality=0.0,
                      source_awareness=0.0, efficiency=0.0, feedback="")

    if step == 0:
        # Step 1 — clarifying question quality
        q = action.query or action.explanation or ""
        rq = _reasoning_score(q, min_words=5)
        r.reasoning_quality = rq
        r.total = _strict_score(rq * 0.2)
        r.feedback = f"Question quality: {rq:.2f}. Good questions narrow down what makes a claim verifiable."

    elif step == 1:
        # Step 2 — search strategy quality
        q = action.query or action.explanation or ""
        source_kws = ["fact-check", "snopes", "reuters", "ap news", "official", "peer-reviewed",
                      "government", "primary source", "academic", "study", "journal"]
        rq = _reasoning_score(q, min_words=8, keywords=source_kws)
        r.reasoning_quality = rq
        r.total = _strict_score(rq * 0.2)
        r.feedback = f"Search strategy score: {rq:.2f}."

    elif step == 2:
        # Step 3 — final verdict
        correct = (action.answer or "").lower().strip() == sample["label"]
        expl = action.explanation or ""
        rq = _reasoning_score(expl, min_words=15)

        r.correctness      = 0.6 if correct else 0.0
        r.reasoning_quality = rq * 0.4
        r.total = _strict_score(r.correctness + r.reasoning_quality)
        r.feedback = (f"{'Correct' if correct else 'Wrong'} verdict. "
                      f"Explanation quality: {rq:.2f}. "
                      f"Total: {r.total:.2f}/1.0")

    return r


def grade_step_hard(step: int, action: MisinfoAction, sample: dict, history: list) -> MisinfoReward:
    r = MisinfoReward(total=0.0, correctness=0.0, reasoning_quality=0.0,
                      source_awareness=0.0, efficiency=0.0, feedback="")

    source_kws = ["source", "author", "website", "journal", "credib", "reliab",
                  "bias", "publication", "outlet", "funded", "peer"]
    cross_kws  = ["contradicts", "confirms", "consistent", "inconsistent", "agrees",
                  "disagrees", "mainstream", "consensus", "disputed", "debunked"]

    if step == 0:
        q = action.query or action.explanation or ""
        rq = _reasoning_score(q, min_words=6)
        r.reasoning_quality = rq
        r.total = _strict_score(rq * 0.1)
        r.feedback = f"Claim identification quality: {rq:.2f}."

    elif step == 1:
        q = action.query or action.explanation or ""
        rq = _reasoning_score(q, min_words=10, keywords=source_kws)
        r.reasoning_quality = rq
        r.total = _strict_score(rq * 0.15)
        r.feedback = f"Search strategy: {rq:.2f}."

    elif step == 2:
        q = action.query or action.explanation or ""
        sa = _reasoning_score(q, min_words=10, keywords=source_kws)
        r.source_awareness = sa
        r.total = _strict_score(sa * 0.15)
        r.feedback = f"Source assessment: {sa:.2f}."

    elif step == 3:
        q = action.query or action.explanation or ""
        rq = _reasoning_score(q, min_words=15, keywords=cross_kws)
        r.reasoning_quality = rq
        r.total = _strict_score(rq * 0.2)
        r.feedback = f"Cross-check quality: {rq:.2f}."

    elif step == 4:
        correct    = (action.answer or "").lower().strip() == sample["label"]
        expl       = action.explanation or ""
        conf       = action.confidence if action.confidence is not None else 0.5
        rq         = _reasoning_score(expl, min_words=30, keywords=source_kws + cross_kws)
        # Calibration bonus: confident AND correct, or uncertain AND wrong
        calib_bonus = 0.1 if (correct and conf >= 0.7) or (not correct and conf < 0.5) else 0.0

        r.correctness       = 0.4 if correct else 0.0
        r.reasoning_quality = rq * 0.3
        r.source_awareness  = 0.1 if any(k in expl.lower() for k in source_kws) else 0.0
        r.efficiency        = calib_bonus

        # Efficiency bonus: if agent was mostly correct across trajectory
        prior_total = sum(h.get("step_reward", 0.0) for h in history)
        if prior_total >= 0.35:
            r.efficiency += 0.1

        r.total = _strict_score(r.correctness + r.reasoning_quality + r.source_awareness + r.efficiency)
        r.feedback = (
            f"Final verdict {'correct' if correct else 'wrong'}. "
            f"Reasoning: {rq:.2f}. Source awareness: {r.source_awareness:.2f}. "
            f"Calibration bonus: {calib_bonus:.2f}. Total: {r.total:.2f}/1.0"
        )

    return r


GRADERS = {
    "easy":   grade_step_easy,
    "medium": grade_step_medium,
    "hard":   grade_step_hard,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MisinfoEnv:
    name = "misinfo-detection"

    def __init__(self, task: str = "easy"):
        assert task in TASK_CONFIG, "task must be easy, medium, or hard"
        self.task        = task
        self.config      = TASK_CONFIG[task]
        self._episode    = 0          # increments on each reset for reproducibility
        self._sample     = None
        self._step_idx   = 0
        self._done       = False
        self._history    = []         # list of {step, action_dict, reward_dict}
        self._last_reward: Optional[MisinfoReward] = None

    # ---- OpenEnv required methods ----------------------------------------

    def reset(self) -> MisinfoObservation:
        self._sample   = _get_sample(self.task, self._episode)
        self._episode += 1
        self._step_idx = 0
        self._done     = False
        self._history  = []
        self._last_reward = None

        return MisinfoObservation(
            task=self.task,
            step=0,
            max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task == "hard" else None,
            prompt=self.config["step_prompts"][0],
            score=0.01,
            done=False,
            feedback="New episode started.",
        )

    def step(self, action: MisinfoAction) -> MisinfoObservation:
        if self._done:
            return self.state()

        grader = GRADERS[self.task]
        reward = grader(self._step_idx, action, self._sample, self._history)
        self._last_reward = reward

        self._history.append({
            "step":        self._step_idx,
            "action":      action.model_dump(),
            "step_reward": reward.total,
        })

        self._step_idx += 1
        is_final = (self._step_idx >= self.config["max_steps"])
        self._done = is_final

        cumulative_score = sum(h["step_reward"] for h in self._history)
        cumulative_score = round(max(0.01, min(0.99, cumulative_score)), 4)

        next_prompt = (
            self.config["step_prompts"][self._step_idx]
            if not is_final
            else "Episode complete."
        )

        return MisinfoObservation(
            task=self.task,
            step=self._step_idx,
            max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task == "hard" else None,
            prompt=next_prompt,
            score=cumulative_score,
            done=is_final,
            feedback=reward.feedback,
        )

    def state(self) -> MisinfoObservation:
        if self._sample is None:
            raise RuntimeError("Call reset() first.")
        raw_cumulative = sum(h["step_reward"] for h in self._history)
        cumulative = _strict_score(raw_cumulative)
        return MisinfoObservation(
            task=self.task,
            step=self._step_idx,
            max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task == "hard" else None,
            prompt=(
                self.config["step_prompts"][self._step_idx]
                if self._step_idx < self.config["max_steps"]
                else "Episode complete."
            ),
            score=cumulative,
            done=self._done,
            feedback=self._last_reward.feedback if self._last_reward else "No steps taken yet.",
        )

    def reward(self) -> MisinfoReward:
        """Returns the last step's detailed reward breakdown."""
        if self._last_reward is None:
            raise RuntimeError("Call step() first.")
        return self._last_reward
