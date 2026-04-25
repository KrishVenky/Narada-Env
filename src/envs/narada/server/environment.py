"""
Narada: Core environment logic (one instance per WebSocket session).

All state is per-instance. Never shared between sessions.
Graph is a module-level singleton loaded once at startup.
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Set

from ..case_generator import MAX_STEPS, PatientCase, generate_case
from ..graph import NaradaGraph, get_graph
from ..models import (
    NaradaAction,
    NaradaObservation,
    NaradaState,
    GraphNode,
    StepResult,
    Variant,
)

logger = logging.getLogger(__name__)

# Validator requires returned scores strictly between 0 and 1 (exclusive).
# Internally we keep signed raw rewards so penalties stay meaningful.
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99
_RAW_SCORE_SCALE = 0.45


def _clamp_score(value: float, default: float = 0.5) -> float:
    if not math.isfinite(value):
        return default
    return float(max(_SCORE_MIN, min(_SCORE_MAX, value)))


def _to_score(raw_reward: float, default: float = 0.5) -> float:
    """Map signed raw reward to OpenEnv's required score interval."""
    if not math.isfinite(raw_reward):
        return default
    return _clamp_score(0.5 + raw_reward * _RAW_SCORE_SCALE, default=default)


# ── Step-level reward constants ───────────────────────────────────────────────

R_RELEVANT_HOP = 0.15
R_IRRELEVANT_HOP = -0.05
R_PER_STEP = -0.01
R_LAB_PENALTY = -0.10
R_BACKTRACK_RECOVERY = 0.05

R_TERMINAL_CORRECT = 1.0
R_TERMINAL_PARTIAL = 0.5      # non-terminal bonus per correct oligogenic flag
R_TERMINAL_WRONG = -0.5
R_TIMING_BONUS = 0.2          # correct flag before the tier's early-step cutoff

OVERSEER_MIN = 0.0
OVERSEER_MAX = 0.3


class NaradaEnvironment:
    """
    Stateful environment for one WebSocket session.
    reset() → step()* → (done)
    """

    def __init__(self) -> None:
        self._graph: NaradaGraph = get_graph()
        self._episode_id: str = ""
        self._case: Optional[PatientCase] = None
        self._step: int = 0
        self._max_steps: int = 0
        self._done: bool = False

        # Navigation state
        self._current_node_id: str = ""
        self._trail: List[str] = []      # node IDs in visit order
        self._trail_set: Set[str] = set()

        # Reward tracking
        self._step_rewards: List[float] = []
        self._raw_cumulative_reward: float = 0.0
        self._cumulative_reward: float = 0.0

        # Flagging state
        self._flagged_allele_ids: List[str] = []

        # Overseer inputs
        self._hallucinated_hops: int = 0   # hops to nodes not in graph edges
        self._reasoning_log: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_type: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
        task_type = (task_type or "monogenic").strip().lower()
        if task_type not in ("monogenic", "oligogenic", "phenotype_mismatch"):
            raise ValueError(f"Unknown task_type: {task_type!r}")

        self._episode_id = str(uuid.uuid4())
        self._case = generate_case(self._graph, task_type, seed=seed)
        self._step = 0
        self._max_steps = MAX_STEPS[task_type]
        self._done = False

        self._current_node_id = self._case.starting_node_id
        self._trail = [self._current_node_id]
        self._trail_set = {self._current_node_id}

        self._step_rewards = []
        self._raw_cumulative_reward = 0.0
        self._cumulative_reward = 0.0
        self._flagged_allele_ids = []
        self._hallucinated_hops = 0
        self._reasoning_log = []

        logger.info(
            "Episode %s | task=%s disease=%s genes=%s",
            self._episode_id[:8], task_type,
            self._case.disease_name, self._case.causal_genes,
        )

        # Reset itself has no reward signal; use the neutral 0.5 score so the
        # value stays strictly within the (0.01, 0.99) range OpenEnv requires
        # for every StepResult.
        neutral_score = _to_score(0.0)
        obs = self._build_observation(step_reward=neutral_score)
        return StepResult(
            observation=obs,
            reward=neutral_score,
            done=False,
            info={"episode_id": self._episode_id},
        )

    def step(self, action: NaradaAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._case is None:
            raise RuntimeError("Call reset() before step().")

        self._step += 1
        if action.reasoning:
            self._reasoning_log.append(action.reasoning[:200])

        # Route to action handler. Values are signed raw rewards until returned.
        step_reward, terminal_reward, terminal = self._dispatch_action(action)

        # Per-step efficiency penalty
        step_reward += R_PER_STEP

        # Terminal condition: timeout
        if self._step >= self._max_steps and not terminal:
            terminal = True
            terminal_reward = self._compute_terminal_reward()

        step_score = _to_score(step_reward)
        self._step_rewards.append(step_score)
        self._raw_cumulative_reward += step_reward
        # Expose a true running mean of the per-step OpenEnv scores. This is
        # what an agent can reason about monotonically; it is NOT the mapped
        # sum of raw rewards (that would be misleading).
        self._cumulative_reward = sum(self._step_rewards) / len(self._step_rewards)

        if terminal:
            # Only successful terminal outcomes receive overseer shaping.
            # Wrong flags and timeouts must remain clearly worse than neutral.
            overseer_score = self._overseer_score() if terminal_reward > 0 else 0.0
            final_reward = _to_score(terminal_reward + overseer_score)
            self._done = True
        else:
            final_reward = step_score

        obs = self._build_observation(step_reward=step_score)

        return StepResult(
            observation=obs,
            reward=final_reward if terminal else step_score,
            done=self._done,
            info={
                "episode_id": self._episode_id,
                "action_type": action.action_type,
                "terminal": terminal,
            },
        )

    def state(self) -> NaradaState:
        return NaradaState(
            episode_id=self._episode_id,
            task_type=self._case.task_type if self._case else "unknown",
            case_id=self._case.case_id if self._case else "",
            step_count=self._step,
            max_steps=self._max_steps,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            flagged_variants=[f"VAR:{aid}" for aid in self._flagged_allele_ids],
            ground_truth_variants=(
                self._case.ground_truth_variant_ids
                if self._case and self._done else []
            ),
        )

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _dispatch_action(
        self, action: NaradaAction
    ) -> tuple[float, float, bool]:
        """Returns (step_reward, terminal_reward, is_terminal)."""
        atype = action.action_type.lower()

        if atype == "hop":
            return self._action_hop(action.node_id or ""), 0.0, False

        if atype == "flag_causal":
            return self._action_flag(action.variant_id or "")

        if atype == "request_lab":
            return R_LAB_PENALTY, 0.0, False

        if atype == "backtrack":
            return self._action_backtrack(), 0.0, False

        if atype == "summarise_trail":
            return 0.0, 0.0, False  # neutral, just informational

        # Unknown action — treat as no-op with small penalty
        return -0.02, 0.0, False

    def _action_hop(self, target_node_id: str) -> float:
        if not target_node_id:
            return R_IRRELEVANT_HOP

        neighbors = self._graph.get_neighbors(self._current_node_id)

        # Hallucination check: node exists but is not connected
        if target_node_id not in neighbors:
            if target_node_id in self._graph.nodes:
                self._hallucinated_hops += 1
            return R_IRRELEVANT_HOP - 0.05

        self._current_node_id = target_node_id
        if target_node_id not in self._trail_set:
            self._trail.append(target_node_id)
            self._trail_set.add(target_node_id)

        # Relevance reward
        is_relevant = target_node_id in self._case.relevant_node_ids
        return R_RELEVANT_HOP if is_relevant else R_IRRELEVANT_HOP

    def _action_backtrack(self) -> float:
        if len(self._trail) < 2:
            return R_IRRELEVANT_HOP

        # Reward backtrack only if last hop was irrelevant
        prev_node = self._current_node_id
        self._trail.pop()
        self._current_node_id = self._trail[-1]

        was_irrelevant = prev_node not in self._case.relevant_node_ids
        return R_BACKTRACK_RECOVERY if was_irrelevant else R_IRRELEVANT_HOP

    def _action_flag(self, variant_id: str) -> tuple[float, float, bool]:
        """Returns (step_reward, terminal_reward, is_terminal)."""
        case = self._case

        # Parse allele_id from variant_id (format: "VAR:12345")
        allele_id = variant_id.replace("VAR:", "").strip()

        # Check if variant is in candidate list at all
        candidate_ids = {v.allele_id for v in case.candidate_variants}
        if allele_id not in candidate_ids:
            return R_TERMINAL_WRONG, R_TERMINAL_WRONG, True

        if allele_id in self._flagged_allele_ids:
            return R_IRRELEVANT_HOP, 0.0, False

        self._flagged_allele_ids.append(allele_id)

        if case.task_type == "oligogenic":
            ground_truth = set(case.causal_allele_ids)
            flagged = set(self._flagged_allele_ids)
            if allele_id not in ground_truth:
                return R_TERMINAL_WRONG, R_TERMINAL_WRONG, True
            if ground_truth.issubset(flagged):
                terminal_reward = self._compute_terminal_reward()
                return R_TERMINAL_PARTIAL, terminal_reward, True
            progress_reward = R_TERMINAL_PARTIAL / max(1, len(ground_truth))
            return progress_reward, 0.0, False

        terminal_reward = self._compute_terminal_reward()
        return terminal_reward * 0.1, terminal_reward, True

    # ── Terminal reward ────────────────────────────────────────────────────────

    def _compute_terminal_reward(self) -> float:
        case = self._case
        ground_truth = set(case.causal_allele_ids)
        flagged = set(self._flagged_allele_ids)

        if not flagged:
            # Timed out without flagging — partial credit based on graph exploration
            exploration_bonus = min(0.2, len(self._trail_set) / max(1, self._max_steps) * 0.25)
            return -0.25 + exploration_bonus

        # Check for decoy flag (phenotype_mismatch task)
        decoy_gene = case.decoy_gene
        if decoy_gene:
            decoy_allele_ids = {
                v["allele_id"]
                for v in self._graph.get_variants_for_gene(decoy_gene)
            }
            if flagged & decoy_allele_ids:
                return R_TERMINAL_WRONG  # Flagged the decoy — maximum penalty

        correct = flagged & ground_truth
        wrong = flagged - ground_truth

        if case.task_type == "monogenic":
            if correct:
                base = R_TERMINAL_CORRECT
                if wrong:
                    base -= 0.3 * len(wrong)
                timing_bonus = R_TIMING_BONUS if self._step < 10 else 0.0
                return base + timing_bonus
            return R_TERMINAL_WRONG

        if case.task_type == "oligogenic":
            n_correct = len(correct)
            n_total = len(ground_truth)
            # Scale oligogenic reward to the same 1.0 ceiling as monogenic so a
            # fully correct diagnosis is not penalised by the tier. Partial
            # credit remains linear in coverage.
            coverage = (n_correct / n_total) if n_total > 0 else 0.0
            partial = coverage * R_TERMINAL_CORRECT
            wrong_penalty = 0.2 * len(wrong)
            timing_bonus = R_TIMING_BONUS if (self._step < 15 and n_correct == n_total) else 0.0
            return partial - wrong_penalty + timing_bonus

        if case.task_type == "phenotype_mismatch":
            if correct:
                timing_bonus = R_TIMING_BONUS if self._step < 12 else 0.0
                return R_TERMINAL_CORRECT + timing_bonus
            return R_TERMINAL_WRONG

        return 0.0

    # ── Overseer ──────────────────────────────────────────────────────────────

    def _overseer_score(self) -> float:
        """
        Additive quality score 0.0–0.3 for the Overseer agent.
        Computed locally (no LLM call in this implementation).
        Full Overseer LLM call can be added in inference.py.
        """
        score = OVERSEER_MAX

        # Penalise hallucinated hops.
        score -= self._hallucinated_hops * 0.05

        # Penalise very short exploration (< 3 unique nodes = no reasoning).
        unique_visited = len(self._trail_set)
        if unique_visited < 3:
            score -= 0.1

        # Reward visiting each causal gene (oligogenic rewards both, capped).
        case = self._case
        gene_bonus = 0.0
        for gene in case.causal_genes:
            if f"GENE:{gene}" in self._trail_set:
                gene_bonus += 0.05
        score += min(0.10, gene_bonus)

        return min(OVERSEER_MAX, max(OVERSEER_MIN, score))

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_observation(self, step_reward: float) -> NaradaObservation:
        case = self._case
        graph = self._graph

        current_node = self._node_to_model(self._current_node_id)
        trail_nodes = [self._node_to_model(nid) for nid in self._trail[-10:]]  # last 10

        info: Dict[str, Any] = {
            "task_type": case.task_type,
            "episode_id": self._episode_id,
            "flagged_variants": [f"VAR:{aid}" for aid in self._flagged_allele_ids],
        }
        if self._done:
            info["disease_name"] = case.disease_name
            info["ground_truth_hint"] = case.causal_genes  # revealed post-episode
            info["ground_truth_variants"] = case.ground_truth_variant_ids

        return NaradaObservation(
            step=self._step,
            max_steps=self._max_steps,
            task_type=case.task_type,
            current_node=current_node,
            trail=trail_nodes,
            patient_phenotypes=case.patient_hpo_ids,
            phenotype_names=case.patient_phenotype_names,
            phenotypes_absent=case.absent_hpo_ids,
            phenotype_absent_names=case.absent_phenotype_names,
            candidate_variants=case.candidate_variants,
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            info=info,
        )

    def _node_to_model(self, node_id: str) -> GraphNode:
        nd = self._graph.get_node(node_id)
        if nd is None:
            return GraphNode(
                id=node_id, type="unknown", name=node_id,
                description="", connected_node_ids=[],
            )
        neighbors = self._graph.get_neighbors(node_id)
        return GraphNode(
            id=nd["id"],
            type=nd["type"],
            name=nd["name"],
            description=nd["description"],
            connected_node_ids=neighbors[:30],
            metadata=nd["metadata"],
        )
