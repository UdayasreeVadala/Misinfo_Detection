"""
Misinformation Detection OpenEnv Environment
=============================================
Implements the full OpenEnv spec: step() / reset() / state()
with three difficulty tiers (easy, medium, hard).
All reward fields are strictly in (0, 1) — never 0.0 or 1.0.
"""

from __future__ import annotations

import os
import re
import time
import random
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ── Score clamping constants ──────────────────────────────────────────
EPSILON_SCORE = 0.01
MAX_SCORE = 0.99


def _strict_score(v: float) -> float:
    """Clamp a float strictly into (0, 1)."""
    return max(EPSILON_SCORE, min(MAX_SCORE, float(v)))


# ── Pydantic Models ──────────────────────────────────────────────────

class MisinfoReward(BaseModel):
    total: float = Field(default=0.01, ge=0.0, le=1.0)
    correctness: float = Field(default=0.01, ge=0.0, le=1.0)
    reasoning_quality: float = Field(default=0.01, ge=0.0, le=1.0)
    source_awareness: float = Field(default=0.01, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.01, ge=0.0, le=1.0)


class MisinfoAction(BaseModel):
    action: str = Field(..., description="Agent action string, e.g. 'classify:real', 'search', 'question', 'verdict:fake'")


class MisinfoObservation(BaseModel):
    text: str = Field(default="", description="The claim or article text to evaluate")
    sources: List[str] = Field(default_factory=list, description="Available or retrieved sources")
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=1, description="Maximum steps for this task")
    task: str = Field(default="easy", description="Current task difficulty")
    done: bool = Field(default=False)
    reward: MisinfoReward = Field(default_factory=lambda: MisinfoReward())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MisinfoState(BaseModel):
    observation: MisinfoObservation
    reward: MisinfoReward
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ── Helper functions (AFTER MisinfoReward is defined) ─────────────────

def _empty_reward() -> MisinfoReward:
    return MisinfoReward(
        total=EPSILON_SCORE,
        correctness=EPSILON_SCORE,
        reasoning_quality=EPSILON_SCORE,
        source_awareness=EPSILON_SCORE,
        efficiency=EPSILON_SCORE,
    )


def _finalize_reward(r: MisinfoReward) -> MisinfoReward:
    """Clamp every field in the reward to strict (0, 1)."""
    return MisinfoReward(
        total=_strict_score(r.total),
        correctness=_strict_score(r.correctness),
        reasoning_quality=_strict_score(r.reasoning_quality),
        source_awareness=_strict_score(r.source_awareness),
        efficiency=_strict_score(r.efficiency),
    )


# ── Sample claims dataset ────────────────────────────────────────────

CLAIMS = {
    "easy": [
        {"text": "The Earth is flat and NASA has been hiding this fact.", "label": "fake",
         "sources": ["NASA official website", "Scientific American"]},
        {"text": "Water boils at 100°C at standard atmospheric pressure.", "label": "real",
         "sources": ["Physics textbook", "NIST standards"]},
        {"text": "5G towers cause COVID-19 infections.", "label": "fake",
         "sources": ["WHO fact-check", "IEEE research paper"]},
        {"text": "The human body contains approximately 60% water.", "label": "real",
         "sources": ["Mayo Clinic", "USGS Water Science"]},
        {"text": "Drinking bleach cures all viral infections.", "label": "fake",
         "sources": ["CDC guidelines", "Poison Control Center"]},
    ],
    "medium": [
        {"text": "A new study shows that moderate coffee consumption reduces the risk of heart disease by 20%.",
         "label": "real",
         "sources": ["American Heart Association", "New England Journal of Medicine", "Reuters Health"],
         "context": "Published in NEJM, peer-reviewed, sample size 50,000"},
        {"text": "Scientists confirm that a common household spice can completely cure diabetes within 30 days.",
         "label": "fake",
         "sources": ["PubMed search results", "FDA advisory", "Snopes fact-check"],
         "context": "Originates from a blog post, no peer review"},
        {"text": "Global average temperatures rose by 1.1°C above pre-industrial levels as of 2023.",
         "label": "real",
         "sources": ["IPCC Report", "NASA GISS", "NOAA Climate"],
         "context": "Confirmed by multiple independent agencies"},
        {"text": "A leaked government memo reveals plans to replace all currency with cryptocurrency by 2025.",
         "label": "fake",
         "sources": ["Federal Reserve statements", "Treasury Department", "Reuters"],
         "context": "No official source confirms, originates from conspiracy forum"},
    ],
    "hard": [
        {"text": "A peer-reviewed study in Nature claims that a specific gene therapy can reverse aging by 25 years. The study was funded by a biotech startup and has not been independently replicated.",
         "label": "fake",
         "sources": ["Nature journal archives", "PubMed", "Retraction Watch", "bioRxiv preprints", "NIH funding database"],
         "context": "While the journal is legitimate, the study has methodological concerns, small sample size (n=8), and the lead author has two previous retractions.",
         "complexity_factors": ["legitimate venue", "real scientific concepts", "funding bias", "replication issues"]},
        {"text": "WHO reports that global malaria deaths decreased by 50% between 2000 and 2020 due to widespread adoption of insecticide-treated bed nets and artemisinin-based therapies.",
         "label": "real",
         "sources": ["WHO World Malaria Report", "The Lancet", "Gates Foundation data", "UNICEF statistics", "CDC global health"],
         "context": "Confirmed by multiple independent health organizations with consistent data across sources.",
         "complexity_factors": ["specific statistics", "long time range", "multiple interventions", "global scope"]},
        {"text": "Recent investigations reveal that a major social media platform deliberately amplified political misinformation to increase engagement, according to a whistleblower with internal documents.",
         "label": "real",
         "sources": ["SEC filings", "Congressional testimony transcripts", "Washington Post investigation", "Internal document leaks", "Platform transparency report"],
         "context": "Multiple corroborating sources including sworn testimony and verified internal documents.",
         "complexity_factors": ["corporate interests", "whistleblower credibility", "document verification", "political sensitivity"]},
    ],
}


# ── Main Environment Class ───────────────────────────────────────────

class MisinfoEnv:
    """
    Misinformation Detection Environment implementing OpenEnv spec.
    Tasks: easy (1 step), medium (3 steps), hard (5 steps).
    """

    TASK_MAX_STEPS = {"easy": 1, "medium": 3, "hard": 5}
    VALID_TASKS = ["easy", "medium", "hard"]

    def __init__(self):
        self._task: str = "easy"
        self._step: int = 0
        self._done: bool = False
        self._claim: Dict[str, Any] = {}
        self._reward: MisinfoReward = _empty_reward()
        self._history: List[str] = []
        self._sources_consulted: List[str] = []

    # ── OpenEnv API ───────────────────────────────────────────────

    def reset(self, task: str = "easy") -> MisinfoState:
        if task not in self.VALID_TASKS:
            task = "easy"
        self._task = task
        self._step = 0
        self._done = False
        self._reward = _empty_reward()
        self._history = []
        self._sources_consulted = []

        claims = CLAIMS.get(task, CLAIMS["easy"])
        self._claim = random.choice(claims)

        return self._build_state()

    def step(self, action: MisinfoAction) -> MisinfoState:
        if self._done:
            return self._build_state()

        self._step += 1
        action_text = action.action.strip().lower()
        self._history.append(action_text)

        # Grade based on task difficulty
        if self._task == "easy":
            self._reward = self._grade_step_easy(action_text)
        elif self._task == "medium":
            self._reward = self._grade_step_medium(action_text)
        else:
            self._reward = self._grade_step_hard(action_text)

        max_steps = self.TASK_MAX_STEPS[self._task]
        if self._step >= max_steps or self._is_terminal_action(action_text):
            self._done = True

        return self._build_state()

    def state(self) -> MisinfoState:
        return self._build_state()

    # ── Grading: Easy ─────────────────────────────────────────────

    def _grade_step_easy(self, action: str) -> MisinfoReward:
        """Easy: single-step classification. Just classify:real or classify:fake."""
        correct_label = self._claim.get("label", "real")
        correctness = EPSILON_SCORE

        if action.startswith("classify:") or action.startswith("verdict:"):
            predicted = action.split(":", 1)[1].strip()
            if predicted == correct_label:
                correctness = 0.95
            else:
                correctness = 0.15

        r = MisinfoReward(
            total=EPSILON_SCORE,
            correctness=correctness,
            reasoning_quality=0.50,
            source_awareness=0.30,
            efficiency=0.90,
        )
        r.total = _strict_score(
            0.50 * r.correctness +
            0.20 * r.reasoning_quality +
            0.15 * r.source_awareness +
            0.15 * r.efficiency
        )
        return _finalize_reward(r)

    # ── Grading: Medium ───────────────────────────────────────────

    def _grade_step_medium(self, action: str) -> MisinfoReward:
        """
        Medium: up to 3 steps. Expects question → search → verdict.
        Partial credit for each meaningful step.
        """
        step = self._step
        correct_label = self._claim.get("label", "real")

        prev = self._reward
        correctness = prev.correctness
        reasoning = prev.reasoning_quality
        source_aw = prev.source_awareness
        efficiency = prev.efficiency

        if action.startswith("question"):
            reasoning = _strict_score(reasoning + 0.20)
        elif action.startswith("search"):
            source_aw = _strict_score(source_aw + 0.25)
            self._sources_consulted.append(action)
        elif action.startswith("classify:") or action.startswith("verdict:"):
            predicted = action.split(":", 1)[1].strip()
            if predicted == correct_label:
                correctness = 0.85
            else:
                correctness = _strict_score(correctness + 0.05)
        else:
            reasoning = _strict_score(reasoning + 0.05)

        # Efficiency bonus: finishing in fewer steps
        max_s = self.TASK_MAX_STEPS[self._task]
        efficiency = _strict_score(0.50 + 0.40 * (1.0 - step / max_s))

        r = MisinfoReward(
            total=EPSILON_SCORE,
            correctness=correctness,
            reasoning_quality=reasoning,
            source_awareness=source_aw,
            efficiency=efficiency,
        )
        r.total = _strict_score(
            0.40 * r.correctness +
            0.25 * r.reasoning_quality +
            0.20 * r.source_awareness +
            0.15 * r.efficiency
        )
        return _finalize_reward(r)

    # ── Grading: Hard ─────────────────────────────────────────────

    def _grade_step_hard(self, action: str) -> MisinfoReward:
        """
        Hard: up to 5 steps. Expects question → search → assess_source → cross_check → verdict.
        """
        step = self._step
        correct_label = self._claim.get("label", "real")

        prev = self._reward
        correctness = prev.correctness
        reasoning = prev.reasoning_quality
        source_aw = prev.source_awareness
        efficiency = prev.efficiency

        if action.startswith("question"):
            reasoning = _strict_score(reasoning + 0.15)
        elif action.startswith("search"):
            source_aw = _strict_score(source_aw + 0.15)
            self._sources_consulted.append(action)
        elif action.startswith("assess_source"):
            source_aw = _strict_score(source_aw + 0.20)
            reasoning = _strict_score(reasoning + 0.10)
        elif action.startswith("cross_check"):
            source_aw = _strict_score(source_aw + 0.15)
            reasoning = _strict_score(reasoning + 0.15)
        elif action.startswith("classify:") or action.startswith("verdict:"):
            predicted = action.split(":", 1)[1].strip()
            if predicted == correct_label:
                correctness = 0.90
            else:
                correctness = _strict_score(correctness + 0.05)
        else:
            reasoning = _strict_score(reasoning + 0.03)

        max_s = self.TASK_MAX_STEPS[self._task]
        efficiency = _strict_score(0.40 + 0.40 * (1.0 - step / max_s))

        r = MisinfoReward(
            total=EPSILON_SCORE,
            correctness=correctness,
            reasoning_quality=reasoning,
            source_awareness=source_aw,
            efficiency=efficiency,
        )
        r.total = _strict_score(
            0.35 * r.correctness +
            0.25 * r.reasoning_quality +
            0.25 * r.source_awareness +
            0.15 * r.efficiency
        )
        return _finalize_reward(r)

    # ── Helpers ───────────────────────────────────────────────────

    def _is_terminal_action(self, action: str) -> bool:
        return action.startswith("classify:") or action.startswith("verdict:")

    def _build_state(self) -> MisinfoState:
        obs = MisinfoObservation(
            text=self._claim.get("text", ""),
            sources=self._claim.get("sources", []),
            step=self._step,
            max_steps=self.TASK_MAX_STEPS[self._task],
            task=self._task,
            done=self._done,
            reward=self._reward,
            metadata={
                "context": self._claim.get("context", ""),
                "complexity_factors": self._claim.get("complexity_factors", []),
                "history": list(self._history),
                "sources_consulted": list(self._sources_consulted),
            },
        )
        return MisinfoState(
            observation=obs,
            reward=self._reward,
            done=self._done,
            info={"task": self._task, "step": self._step},
        )
