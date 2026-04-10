"""
Misinformation Detection Environment
======================================
OpenEnv-compatible RL environment — 3 tasks: easy / medium / hard
All scores strictly in (0.01, 0.99) as required by OpenEnv spec.
"""

import random
import hashlib
from datasets import load_dataset
from pydantic import BaseModel
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# Typed Models
# ---------------------------------------------------------------------------

class MisinfoAction(BaseModel):
    action_type: Literal["classify", "question", "search", "assess_source", "cross_check", "verdict"]
    answer:      Optional[str]   = None
    query:       Optional[str]   = None
    explanation: Optional[str]   = None
    confidence:  Optional[float] = None


class MisinfoObservation(BaseModel):
    task:         str
    step:         int
    max_steps:    int
    article_text: str
    source:       Optional[str] = None
    prompt:       str
    score:        float
    done:         bool
    feedback:     str


class MisinfoReward(BaseModel):
    total:              float
    correctness:        float
    reasoning_quality:  float
    source_awareness:   float
    efficiency:         float
    feedback:           str


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

print("Loading dataset...")
_dataset  = load_dataset("GonzaloA/fake_news", split="train")
_real     = [x for x in _dataset if x["label"] == 1]
_fake     = [x for x in _dataset if x["label"] == 0]
random.shuffle(_real); random.shuffle(_fake)
REAL_POOL = _real[:500]
FAKE_POOL = _fake[:500]
ALL_POOL  = REAL_POOL + FAKE_POOL
print(f"Dataset ready — real: {len(REAL_POOL)}, fake: {len(FAKE_POOL)}")


def _clamp(v: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return round(max(0.01, min(0.99, v)), 2)


def _get_sample(task: str, episode: int) -> dict:
    idx    = int(hashlib.md5(f"{task}-{episode}".encode()).hexdigest(), 16) % len(ALL_POOL)
    sample = ALL_POOL[idx]
    label  = "real" if sample["label"] == 1 else "fake"
    full   = sample.get("text", "") or ""
    if task == "easy":
        text = sample["title"] or full[:120]
    elif task == "medium":
        text = full[:400] if full else sample["title"]
    else:
        text = full[:800] if full else sample["title"]
    return {"title": sample.get("title",""), "text": text,
            "label": label, "source": sample.get("subject","unknown"), "full": full}


TASK_CONFIG = {
    "easy":   {"max_steps": 1, "step_prompts": [
        "Read the headline and classify it as 'real' or 'fake'. Use action_type='classify'."]},
    "medium": {"max_steps": 3, "step_prompts": [
        "Read the article snippet. Use action_type='question' to ask one clarifying question.",
        "Use action_type='search' to describe what evidence you would look for.",
        "Give your final verdict. Use action_type='verdict' with answer='real' or 'fake' and explanation."]},
    "hard":   {"max_steps": 5, "step_prompts": [
        "Use action_type='question' to identify the most suspicious claim.",
        "Use action_type='search' to describe your search strategy.",
        "Use action_type='assess_source' to evaluate source credibility.",
        "Use action_type='cross_check' to compare with credible sources.",
        "Give final verdict. Use action_type='verdict' with answer, confidence 0-1, and explanation."]},
}

# ---------------------------------------------------------------------------
# Graders — all scores via _clamp() so always in (0.01, 0.99)
# ---------------------------------------------------------------------------

def _reasoning_score(text: str, min_words: int, keywords: list = None) -> float:
    if not text:
        return 0.0
    words = text.lower().split()
    score = 0.0
    if len(words) >= min_words:
        score += 0.5
    elif len(words) >= min_words // 2:
        score += 0.25
    logic = ["because","therefore","however","suggests","indicates","evidence",
             "appears","likely","based on","consistent","inconsistent"]
    if any(m in text.lower() for m in logic):
        score += 0.3
    if keywords and any(k in text.lower() for k in keywords):
        score += 0.2
    return min(score, 0.99)


def grade_easy(step, action, sample, history):
    correct = (action.answer or "").lower().strip() == sample["label"]
    score   = _clamp(0.95 if correct else 0.05)
    return MisinfoReward(total=score, correctness=score,
        reasoning_quality=0.0, source_awareness=0.0, efficiency=0.0,
        feedback="Correct!" if correct else f"Wrong — it was {sample['label']}.")


def grade_medium(step, action, sample, history):
    if step == 0:
        rq = _reasoning_score(action.query or action.explanation or "", 5)
        return MisinfoReward(total=_clamp(rq*0.2+0.05), correctness=0.0,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"Question quality: {rq:.2f}.")
    elif step == 1:
        src = ["fact-check","snopes","reuters","official","government","academic","study"]
        rq  = _reasoning_score(action.query or action.explanation or "", 8, src)
        return MisinfoReward(total=_clamp(rq*0.2+0.05), correctness=0.0,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"Search strategy: {rq:.2f}.")
    else:
        correct = (action.answer or "").lower().strip() == sample["label"]
        rq      = _reasoning_score(action.explanation or "", 15)
        c_score = 0.55 if correct else 0.05
        total   = _clamp(c_score + rq*0.4)
        return MisinfoReward(total=total, correctness=c_score,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"{'Correct' if correct else 'Wrong'} verdict. Explanation: {rq:.2f}. Total: {total:.2f}")


def grade_hard(step, action, sample, history):
    src_kw  = ["source","author","journal","credib","reliab","bias","publication","outlet"]
    cross_kw= ["contradicts","confirms","consistent","inconsistent","consensus","disputed","debunked"]
    text    = action.query or action.explanation or ""

    if step == 0:
        rq = _reasoning_score(text, 6)
        return MisinfoReward(total=_clamp(rq*0.1+0.05), correctness=0.0,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"Claim identification: {rq:.2f}.")
    elif step == 1:
        rq = _reasoning_score(text, 10, src_kw)
        return MisinfoReward(total=_clamp(rq*0.15+0.05), correctness=0.0,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"Search strategy: {rq:.2f}.")
    elif step == 2:
        sa = _reasoning_score(text, 10, src_kw)
        return MisinfoReward(total=_clamp(sa*0.15+0.05), correctness=0.0,
            reasoning_quality=0.0, source_awareness=sa, efficiency=0.0,
            feedback=f"Source assessment: {sa:.2f}.")
    elif step == 3:
        rq = _reasoning_score(text, 15, cross_kw)
        return MisinfoReward(total=_clamp(rq*0.2+0.05), correctness=0.0,
            reasoning_quality=rq, source_awareness=0.0, efficiency=0.0,
            feedback=f"Cross-check: {rq:.2f}.")
    else:
        correct = (action.answer or "").lower().strip() == sample["label"]
        expl    = action.explanation or ""
        conf    = action.confidence if action.confidence is not None else 0.5
        rq      = _reasoning_score(expl, 30, src_kw+cross_kw)
        calib   = 0.08 if (correct and conf >= 0.7) or (not correct and conf < 0.5) else 0.0
        c_score = 0.38 if correct else 0.05
        sa      = 0.08 if any(k in expl.lower() for k in src_kw) else 0.0
        prior   = sum(h.get("step_reward",0.0) for h in history)
        eff     = 0.08 if prior >= 0.3 else 0.0
        total   = _clamp(c_score + rq*0.3 + sa + calib + eff)
        return MisinfoReward(total=total, correctness=c_score,
            reasoning_quality=rq, source_awareness=sa, efficiency=eff+calib,
            feedback=f"{'Correct' if correct else 'Wrong'} verdict. Total: {total:.2f}")


GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MisinfoEnv:
    name = "misinfo-detection"

    def __init__(self, task: str = "easy"):
        assert task in TASK_CONFIG
        self.task          = task
        self.config        = TASK_CONFIG[task]
        self._episode      = 0
        self._sample       = None
        self._step_idx     = 0
        self._done         = False
        self._history      = []
        self._last_reward  = None

    def reset(self) -> MisinfoObservation:
        self._sample   = _get_sample(self.task, self._episode)
        self._episode += 1
        self._step_idx = 0
        self._done     = False
        self._history  = []
        self._last_reward = None
        return MisinfoObservation(
            task=self.task, step=0, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=self.config["step_prompts"][0],
            score=0.05, done=False, feedback="New episode started.")

    def step(self, action: MisinfoAction) -> MisinfoObservation:
        if self._done:
            return self.state()
        reward = GRADERS[self.task](self._step_idx, action, self._sample, self._history)
        self._last_reward = reward
        self._history.append({"step": self._step_idx, "action": action.model_dump(),
                               "step_reward": reward.total})
        self._step_idx += 1
        is_final = self._step_idx >= self.config["max_steps"]
        self._done = is_final
        cumulative = _clamp(sum(h["step_reward"] for h in self._history))
        next_prompt = (self.config["step_prompts"][self._step_idx]
                       if not is_final else "Episode complete.")
        return MisinfoObservation(
            task=self.task, step=self._step_idx, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=next_prompt, score=cumulative, done=is_final, feedback=reward.feedback)

    def state(self) -> MisinfoObservation:
        if self._sample is None:
            raise RuntimeError("Call reset() first.")
        cumulative = _clamp(sum(h["step_reward"] for h in self._history)) if self._history else 0.05
        return MisinfoObservation(
            task=self.task, step=self._step_idx, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=(self.config["step_prompts"][self._step_idx]
                    if self._step_idx < self.config["max_steps"] else "Episode complete."),
            score=cumulative, done=self._done,
            feedback=self._last_reward.feedback if self._last_reward else "No steps yet.")

    def reward(self) -> MisinfoReward:
        if self._last_reward is None:
            raise RuntimeError("Call step() first.")
        return self._last_reward
