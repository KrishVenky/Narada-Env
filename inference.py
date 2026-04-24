"""
ClinDetect: Inference Script (OpenEnv compliant)

Required environment variables:
    API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key — mandatory, no default

Optional:
    ENV_URL        ClinDetect space URL (default: https://krishvenky-clindetect-env.hf.space)
    MAX_STEPS      Override per-episode step limit
    GROQ_BASE_URL  Use Groq for fast dev inference (https://api.groq.com/openai/v1)

Output format (exact — validator parses these lines):
    [START] task=<name> env=clindetect model=<model>
    [STEP]  step=N action=<str> reward=R done=false|true error=null|<msg>
    [END]   success=true|false steps=N score=0.XXX rewards=r1,r2,...
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_URL: str = os.getenv("ENV_URL", "https://krishvenky-clindetect-env.hf.space")
MAX_STEPS_OVERRIDE: Optional[int] = int(os.getenv("MAX_STEPS", "0")) or None

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Inject src path so imports work when running from repo root ──────────────

_src = os.path.join(os.path.dirname(__file__), "src", "envs")
if _src not in sys.path:
    sys.path.insert(0, _src)

from clindetect.client import ClinDetectEnv
from clindetect.models import ClinDetectAction, ClinDetectObservation, StepResult

# ── OpenAI-compat client ──────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TEMPERATURE: float = 0.2
MAX_TOKENS: int = 800

# ── Score clamping ─────────────────────────────────────────────────────────────

def clamp_open_score(value: float, low: float = 0.01, high: float = 0.99, default: float = 0.5) -> float:
    """Clamp to open interval (0,1). Guards NaN/inf."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not math.isfinite(numeric):
        numeric = default
    return max(low, min(high, numeric))


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert clinical geneticist navigating a gene-disease knowledge graph
to diagnose a rare disease patient.

SITUATION:
  - The patient has a set of HPO phenotype terms (presenting symptoms).
  - You must navigate the graph to find the causal variant(s) from a candidate pool.
  - You observe only your current graph node and the nodes you have visited.
  - The graph contains: phenotype nodes, disease nodes, gene nodes, variant nodes, pathway nodes.

GRAPH NAVIGATION:
  - hop(node_id)         : Move to a directly connected node. You see its name, type, neighbors.
  - flag_causal(variant_id) : Declare your diagnosis. This ends the episode.
  - backtrack()          : Return to the previous node. Use after a wrong-direction chain.
  - request_lab(test)    : Get additional clinical data. Penalised — only if essential.
  - summarise_trail()    : Receive a summary of all nodes visited so far.

CLINICAL REASONING RULES:
  1. Always start from phenotype nodes — map symptoms to diseases.
  2. Follow the phenotype → disease → gene → variant path.
  3. Pathogenicity score is NOT sufficient — the variant must match the patient's phenotypes.
  4. A high-pathogenicity BRCA1/TP53/cancer gene variant is a DECOY if phenotypes are cardiac
     or neurological. Resist it.
  5. In oligogenic cases: you must flag ALL contributing variants for full reward.
  6. Efficiency matters — correct early flags earn a timing bonus.

OUTPUT FORMAT (strict JSON):
{
  "action_type": "hop" | "flag_causal" | "backtrack" | "request_lab" | "summarise_trail",
  "node_id": "<node id for hop, omit otherwise>",
  "variant_id": "<VAR:xxxxx for flag_causal, omit otherwise>",
  "test_type": "<test name for request_lab, omit otherwise>",
  "reasoning": "<one sentence of clinical reasoning>"
}
""").strip()


# ── Observation formatter ─────────────────────────────────────────────────────

def format_observation(obs: ClinDetectObservation) -> str:
    lines = [
        f"STEP {obs.step}/{obs.max_steps} | Task: {obs.task_type}",
        "",
        "PATIENT PHENOTYPES:",
    ]
    for hpo_id, name in zip(obs.patient_phenotypes, obs.phenotype_names):
        lines.append(f"  {hpo_id} — {name}")

    lines += ["", f"CURRENT NODE: [{obs.current_node.type.upper()}] {obs.current_node.name}"]
    lines.append(f"  ID: {obs.current_node.id}")
    lines.append(f"  Description: {obs.current_node.description[:100]}")
    lines.append(f"  Connected nodes ({len(obs.current_node.connected_node_ids)} total):")
    for nid in obs.current_node.connected_node_ids[:8]:
        lines.append(f"    {nid}")
    if len(obs.current_node.connected_node_ids) > 8:
        lines.append(f"    ... and {len(obs.current_node.connected_node_ids) - 8} more")

    if obs.trail:
        lines.append(f"\nTRAIL ({len(obs.trail)} nodes visited):")
        for node in obs.trail[-5:]:
            lines.append(f"  [{node.type}] {node.name} ({node.id})")

    lines += ["", "CANDIDATE VARIANTS (choose one to flag_causal):"]
    for v in obs.candidate_variants:
        lines.append(
            f"  {v.id} | {v.gene} | {v.variant_type} | "
            f"pathogenicity={v.pathogenicity_score:.2f} | "
            f"significance={v.clinical_significance}"
        )
        if v.disease_associations:
            lines.append(f"    diseases: {', '.join(v.disease_associations[:2])}")

    lines.append(f"\nStep reward: {obs.step_reward:+.4f} | Cumulative: {obs.cumulative_reward:.4f}")
    lines.append("\nRespond with JSON action.")
    return "\n".join(lines)


# ── Action parser ─────────────────────────────────────────────────────────────

_FALLBACK_ACTION = ClinDetectAction(
    action_type="summarise_trail",
    reasoning="Fallback: gathering information.",
)


def parse_action(text: str) -> ClinDetectAction:
    if not text:
        return _FALLBACK_ACTION
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return _FALLBACK_ACTION
    try:
        data = json.loads(match.group(0))
        atype = str(data.get("action_type", "summarise_trail")).lower()
        if atype not in ("hop", "flag_causal", "backtrack", "request_lab", "summarise_trail"):
            atype = "summarise_trail"
        return ClinDetectAction(
            action_type=atype,
            node_id=str(data["node_id"]) if data.get("node_id") else None,
            variant_id=str(data["variant_id"]) if data.get("variant_id") else None,
            test_type=str(data.get("test_type", "")) or None,
            reasoning=str(data.get("reasoning", ""))[:300],
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return _FALLBACK_ACTION


def action_to_str(action: ClinDetectAction) -> str:
    if action.action_type == "hop" and action.node_id:
        return f"hop({action.node_id})"
    if action.action_type == "flag_causal" and action.variant_id:
        return f"flag_causal({action.variant_id})"
    if action.action_type == "request_lab":
        return f"request_lab({action.test_type or ''})"
    return action.action_type


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(task_type: str) -> None:
    step_rewards: List[float] = []
    steps_taken = 0
    score = 0.5
    success = False

    print(f"[START] task={task_type} env=clindetect model={MODEL_NAME}", flush=True)

    try:
        async with ClinDetectEnv(base_url=ENV_URL) as env:
            result = await env.reset(task_type=task_type)
            obs = result.observation

            max_steps = MAX_STEPS_OVERRIDE or obs.max_steps
            conversation: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not obs.done and steps_taken < max_steps:
                steps_taken += 1
                user_content = format_observation(obs)
                conversation.append({"role": "user", "content": user_content})

                action = _FALLBACK_ACTION
                response_text = ""
                for attempt in range(3):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=conversation,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                        )
                        response_text = completion.choices[0].message.content or ""
                        parsed = parse_action(response_text)
                        if parsed is not _FALLBACK_ACTION or attempt == 2:
                            action = parsed
                            conversation.append({"role": "assistant", "content": response_text})
                            break
                    except Exception:
                        if attempt == 2:
                            break

                error_str = "null"
                try:
                    result = await asyncio.wait_for(env.step(action), timeout=30.0)
                except asyncio.TimeoutError:
                    error_str = "timeout"
                    step_rewards.append(0.01)
                    print(
                        f"[STEP] step={steps_taken} action={action_to_str(action)} "
                        f"reward=0.01 done=false error={error_str}",
                        flush=True,
                    )
                    break

                obs = result.observation
                raw_reward = result.reward if obs.done else obs.step_reward
                reward = clamp_open_score(raw_reward)
                step_rewards.append(reward)
                done_str = "true" if obs.done else "false"

                print(
                    f"[STEP] step={steps_taken} action={action_to_str(action)} "
                    f"reward={reward:.2f} done={done_str} error={error_str}",
                    flush=True,
                )

                if obs.done:
                    score = clamp_open_score(result.reward)
                    success = score > 0.5
                    break

    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:100]
        print(
            f"[STEP] step={steps_taken} action=error reward=0.01 done=false error={error_msg}",
            flush=True,
        )

    finally:
        safe_score = clamp_open_score(score)
        safe_rewards = [clamp_open_score(r) for r in step_rewards] if step_rewards else [0.50]
        rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={safe_score:.3f} rewards={rewards_str}",
            flush=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    tasks = [
        "monogenic",
        "oligogenic",
        "phenotype_mismatch",
    ]
    for task_type in tasks:
        await run_episode(task_type)


if __name__ == "__main__":
    asyncio.run(main())
