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
    total:             float
    correctness:       float
    reasoning_quality: float
    source_awareness:  float
    efficiency:        float
    feedback:          str

print("Loading dataset...")
_dataset  = load_dataset("GonzaloA/fake_news", split="train")
_real     = [x for x in _dataset if x["label"] == 1]
_fake     = [x for x in _dataset if x["label"] == 0]
REAL_POOL = _real[:500]
FAKE_POOL = _fake[:500]
ALL_POOL  = REAL_POOL + FAKE_POOL
print(f"Dataset ready — real: {len(REAL_POOL)}, fake: {len(FAKE_POOL)}")

def S(v):
    """Strictly between 0 and 1"""
    return round(max(0.01, min(0.97, float(v))), 4)

def _get_sample(task, episode):
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
    return {"text": text, "label": label, "source": sample.get("subject","unknown")}

TASK_CONFIG = {
    "easy":   {"max_steps": 1, "step_prompts": [
        "Classify as 'real' or 'fake'. Use action_type='classify'."]},
    "medium": {"max_steps": 3, "step_prompts": [
        "Use action_type='question' to ask a clarifying question.",
        "Use action_type='search' to describe your search strategy.",
        "Use action_type='verdict' with answer and explanation."]},
    "hard":   {"max_steps": 5, "step_prompts": [
        "Use action_type='question' to identify suspicious claim.",
        "Use action_type='search' for your search strategy.",
        "Use action_type='assess_source' to evaluate credibility.",
        "Use action_type='cross_check' to compare with credible sources.",
        "Use action_type='verdict' with answer, confidence and explanation."]},
}

def _rscore(text, min_words, kws=None):
    if not text: return 0.05
    words = text.lower().split()
    sc = 0.05
    if len(words) >= min_words: sc += 0.45
    elif len(words) >= min_words//2: sc += 0.20
    logic = ["because","therefore","however","suggests","indicates","evidence","based on"]
    if any(m in text.lower() for m in logic): sc += 0.25
    if kws and any(k in text.lower() for k in kws): sc += 0.20
    return S(sc)

def grade_easy(step, action, sample, history):
    correct = (action.answer or "").lower().strip() == sample["label"]
    score   = S(0.95 if correct else 0.05)
    return MisinfoReward(total=score, correctness=score,
        reasoning_quality=0.01, source_awareness=0.01, efficiency=0.01,
        feedback="Correct!" if correct else f"Wrong — it was {sample['label']}.")

def grade_medium(step, action, sample, history):
    if step == 0:
        rq = _rscore(action.query or action.explanation or "", 5)
        return MisinfoReward(total=S(rq*0.2+0.05), correctness=0.01,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"Question quality: {rq:.2f}.")
    elif step == 1:
        src = ["fact-check","snopes","reuters","official","government","academic","study"]
        rq  = _rscore(action.query or action.explanation or "", 8, src)
        return MisinfoReward(total=S(rq*0.2+0.05), correctness=0.01,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"Search strategy: {rq:.2f}.")
    else:
        correct = (action.answer or "").lower().strip() == sample["label"]
        rq      = _rscore(action.explanation or "", 15)
        c       = S(0.55 if correct else 0.05)
        total   = S(c + rq*0.35)
        return MisinfoReward(total=total, correctness=c,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"{'Correct' if correct else 'Wrong'} verdict. Total: {total:.2f}")

def grade_hard(step, action, sample, history):
    src_kw   = ["source","author","journal","credib","reliab","bias","publication"]
    cross_kw = ["contradicts","confirms","consistent","inconsistent","consensus","disputed"]
    text     = action.query or action.explanation or ""
    if step == 0:
        rq = _rscore(text, 6)
        return MisinfoReward(total=S(rq*0.1+0.05), correctness=0.01,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"Claim identification: {rq:.2f}.")
    elif step == 1:
        rq = _rscore(text, 10, src_kw)
        return MisinfoReward(total=S(rq*0.15+0.05), correctness=0.01,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"Search: {rq:.2f}.")
    elif step == 2:
        sa = _rscore(text, 10, src_kw)
        return MisinfoReward(total=S(sa*0.15+0.05), correctness=0.01,
            reasoning_quality=0.01, source_awareness=sa, efficiency=0.01,
            feedback=f"Source assessment: {sa:.2f}.")
    elif step == 3:
        rq = _rscore(text, 15, cross_kw)
        return MisinfoReward(total=S(rq*0.2+0.05), correctness=0.01,
            reasoning_quality=rq, source_awareness=0.01, efficiency=0.01,
            feedback=f"Cross-check: {rq:.2f}.")
    else:
        correct = (action.answer or "").lower().strip() == sample["label"]
        expl    = action.explanation or ""
        conf    = action.confidence if action.confidence is not None else 0.5
        rq      = _rscore(expl, 30, src_kw+cross_kw)
        calib   = 0.07 if (correct and conf >= 0.7) or (not correct and conf < 0.5) else 0.01
        c       = S(0.38 if correct else 0.05)
        sa      = S(0.08 if any(k in expl.lower() for k in src_kw) else 0.01)
        total   = S(c + rq*0.28 + sa + calib)
        return MisinfoReward(total=total, correctness=c,
            reasoning_quality=rq, source_awareness=sa, efficiency=calib,
            feedback=f"{'Correct' if correct else 'Wrong'} verdict. Total: {total:.2f}")

GRADERS = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

class MisinfoEnv:
    name = "misinfo-detection"

    def __init__(self, task="easy"):
        assert task in TASK_CONFIG
        self.task         = task
        self.config       = TASK_CONFIG[task]
        self._episode     = 0
        self._sample      = None
        self._step_idx    = 0
        self._done        = False
        self._history     = []
        self._last_reward = None

    def reset(self):
        self._sample      = _get_sample(self.task, self._episode)
        self._episode    += 1
        self._step_idx    = 0
        self._done        = False
        self._history     = []
        self._last_reward = None
        return MisinfoObservation(
            task=self.task, step=0, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=self.config["step_prompts"][0],
            score=0.01, done=False, feedback="New episode started.")

    def step(self, action):
        if self._done:
            return self.state()
        reward = GRADERS[self.task](self._step_idx, action, self._sample, self._history)
        self._last_reward = reward
        self._history.append({"step": self._step_idx, "step_reward": reward.total})
        self._step_idx += 1
        is_final   = self._step_idx >= self.config["max_steps"]
        self._done = is_final
        cumulative = S(sum(h["step_reward"] for h in self._history))
        next_prompt = (self.config["step_prompts"][self._step_idx]
                       if not is_final else "Episode complete.")
        return MisinfoObservation(
            task=self.task, step=self._step_idx, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=next_prompt, score=cumulative, done=is_final, feedback=reward.feedback)

    def state(self):
        if self._sample is None:
            raise RuntimeError("Call reset() first.")
        cumulative = S(sum(h["step_reward"] for h in self._history)) if self._history else 0.01
        return MisinfoObservation(
            task=self.task, step=self._step_idx, max_steps=self.config["max_steps"],
            article_text=self._sample["text"],
            source=self._sample["source"] if self.task=="hard" else None,
            prompt=(self.config["step_prompts"][self._step_idx]
                    if self._step_idx < self.config["max_steps"] else "Episode complete."),
            score=cumulative, done=self._done,
            feedback=self._last_reward.feedback if self._last_reward else "No steps yet.")

    def reward(self):
        if self._last_reward is None:
            raise RuntimeError("Call step() first.")
        return self._last_reward
