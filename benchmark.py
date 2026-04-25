"""
Narada Multi-Model Benchmark

Runs a grid of models × tasks × seeds via HF Inference Router and prints a
comparison table. All models are under 5B parameters — trainable on T4.

Usage:
    HF_TOKEN=hf_... python benchmark.py
    HF_TOKEN=hf_... python benchmark.py --n_seeds 5 --tasks monogenic,oligogenic

Output:
    - Console table (markdown)
    - benchmark_results.json  (embed in README)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import websockets

# ── Models under 5B ──────────────────────────────────────────────────────────

MODELS = [
    # (display_name, model_id, backend)
    ("Qwen3-0.6B",      "Qwen/Qwen3-0.6B",                    "hf"),
    ("Qwen3-1.7B",      "Qwen/Qwen3-1.7B",                    "hf"),
    ("Qwen3-4B",        "Qwen/Qwen3-4B",                      "hf"),
    ("Llama-3.2-3B",    "meta-llama/Llama-3.2-3B-Instruct",   "hf"),
    ("Phi-3.5-mini",    "microsoft/Phi-3.5-mini-instruct",     "hf"),
]

TASKS = ["monogenic", "oligogenic", "phenotype_mismatch"]
ENV_URL = os.getenv("ENV_URL", "https://krishvenky-narada-env.hf.space")
HF_BASE = "https://router.huggingface.co/v1"
TEMPERATURE = 0.3
MAX_TOKENS = 600

SYSTEM_PROMPT = """You are a clinical geneticist navigating a gene-disease knowledge graph.
Navigate phenotype -> disease -> gene -> variant.
BRCA1/TP53 is a DECOY if phenotypes are cardiac/neurological.
Output strict JSON:
{"action_type": "hop"|"flag_causal"|"backtrack"|"summarise_trail", "node_id": "<id>", "variant_id": "<VAR:xxxxx>", "reasoning": "<one sentence>"}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def clamp(v: float) -> float:
    if not math.isfinite(v):
        return 0.5
    return max(0.01, min(0.99, float(v)))


def parse_action(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action_type": "summarise_trail", "reasoning": "fallback"}
    try:
        d = json.loads(m.group(0))
        atype = str(d.get("action_type", "summarise_trail")).lower()
        if atype not in ("hop", "flag_causal", "backtrack", "summarise_trail"):
            atype = "summarise_trail"
        return {
            "action_type": atype,
            "node_id": str(d["node_id"]) if d.get("node_id") else None,
            "variant_id": str(d["variant_id"]) if d.get("variant_id") else None,
            "reasoning": str(d.get("reasoning", ""))[:200],
        }
    except Exception:
        return {"action_type": "summarise_trail", "reasoning": "parse error"}


def format_obs(obs: Dict[str, Any]) -> str:
    lines = [f"STEP {obs['step']}/{obs['max_steps']} | {obs['task_type']}"]
    lines.append("PHENOTYPES: " + ", ".join(obs.get("phenotype_names", [])))
    absent = obs.get("phenotype_absent_names") or []
    if absent:
        lines.append("ABSENT: " + ", ".join(absent))
    n = obs["current_node"]
    lines.append(f"NODE: [{n['type']}] {n['name']} ({n['id']})")
    lines.append(f"NEIGHBORS: {', '.join(n['connected_node_ids'][:6])}")
    lines.append("VARIANTS: " + " | ".join(
        f"{v['id']} {v['gene']} path={v['pathogenicity_score']:.2f}"
        for v in obs["candidate_variants"]
    ))
    lines.append(f"step_reward={obs['step_reward']:+.3f}")
    return "\n".join(lines)


# ── Episode runner ─────────────────────────────────────────────────────────────

async def run_episode_async(
    client: OpenAI,
    model_id: str,
    task_type: str,
    seed: int,
    max_steps: int = 15,
) -> Dict[str, Any]:
    ws_url = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    steps_taken = 0
    final_score = 0.5
    success = False
    step_log: List[Dict] = []

    try:
        async with websockets.connect(ws_url, open_timeout=30, ping_interval=20) as ws:
            await ws.send(json.dumps({"type": "reset", "task_type": task_type, "seed": seed}))
            raw = json.loads(await ws.recv())
            if raw.get("type") == "error":
                return {"score": 0.1, "steps": 0, "success": False, "log": []}

            obs = raw["data"]["observation"]
            conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not obs.get("done") and steps_taken < max_steps:
                steps_taken += 1
                conversation.append({"role": "user", "content": format_obs(obs)})

                try:
                    resp = client.chat.completions.create(
                        model=model_id,
                        messages=conversation,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    text = resp.choices[0].message.content or ""
                except Exception as e:
                    text = ""
                    print(f"  [LLM error] {e}", flush=True)

                action = parse_action(text)
                conversation.append({"role": "assistant", "content": text})

                await ws.send(json.dumps({"type": "step", "action": action}))
                raw = json.loads(await ws.recv())
                if raw.get("type") == "error":
                    break

                data = raw["data"]
                obs = data["observation"]
                reward = clamp(data["reward"])
                step_log.append({"step": steps_taken, "action": action["action_type"], "reward": reward})

                if obs.get("done"):
                    final_score = clamp(data["reward"])
                    success = final_score >= 0.70
                    break

    except Exception as e:
        print(f"  [episode error] {e}", flush=True)

    return {
        "score": final_score,
        "steps": steps_taken,
        "success": success,
        "log": step_log,
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

async def benchmark_model(
    display_name: str,
    model_id: str,
    tasks: List[str],
    seeds: List[int],
    hf_token: str,
) -> Dict[str, Any]:
    client = OpenAI(base_url=HF_BASE, api_key=hf_token)
    results: Dict[str, List] = {t: [] for t in tasks}

    for task in tasks:
        print(f"  [{display_name}] {task} ...", flush=True)
        for seed in seeds:
            r = await run_episode_async(client, model_id, task, seed)
            results[task].append(r)
            print(f"    seed={seed} score={r['score']:.3f} steps={r['steps']} ok={r['success']}", flush=True)
            await asyncio.sleep(0.5)  # gentle rate limit

    summary: Dict[str, Any] = {"model": display_name, "model_id": model_id}
    for task in tasks:
        scores = [r["score"] for r in results[task]]
        successes = [r["success"] for r in results[task]]
        avg_steps = sum(r["steps"] for r in results[task]) / max(1, len(results[task]))
        summary[task] = {
            "avg_score": round(sum(scores) / len(scores), 3),
            "success_rate": round(sum(successes) / len(successes), 2),
            "avg_steps": round(avg_steps, 1),
        }
    return summary


def print_table(summaries: List[Dict], tasks: List[str]) -> None:
    header = f"{'Model':<18}" + "".join(f"  {t[:10]:<12}" for t in tasks)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for s in summaries:
        row = f"{s['model']:<18}"
        for t in tasks:
            if t in s:
                d = s[t]
                row += f"  {d['avg_score']:.3f}({d['success_rate']:.0%})  "
        print(row)
    print("=" * len(header))
    print("Format: avg_score(success_rate%)")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    tasks = args.tasks.split(",") if args.tasks else TASKS
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    models = [(n, m, b) for n, m, b in MODELS if args.models == "all" or n in args.models.split(",")]

    print(f"Benchmark: {len(models)} models × {len(tasks)} tasks × {len(seeds)} seeds")
    print(f"Models: {[n for n,_,_ in models]}")
    print(f"Tasks:  {tasks}")
    print(f"Seeds:  {seeds}\n")

    all_results = []
    for display_name, model_id, _ in models:
        print(f"\n── {display_name} ({model_id}) ──")
        t0 = time.time()
        summary = await benchmark_model(display_name, model_id, tasks, seeds, hf_token)
        summary["elapsed_s"] = round(time.time() - t0, 1)
        all_results.append(summary)

    print_table(all_results, tasks)

    out_path = "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks, "seeds": seeds, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Narada multi-model benchmark")
    parser.add_argument("--models", default="all", help="Comma-separated model display names, or 'all'")
    parser.add_argument("--tasks", default="", help="Comma-separated tasks (default: all 3)")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per task per model")
    parser.add_argument("--seed_start", type=int, default=42, help="Starting seed")
    asyncio.run(main(parser.parse_args()))
