"""
Narada GRPO Training Script
Equivalent to narada_grpo.ipynb but runs as a single process.

Required env vars:
    HF_TOKEN       — HuggingFace write token
    HF_PUSH_REPO   — where to push adapter, e.g. "KrishVenky/narada-detective-lora"

Optional env vars:
    ENV_URL        — Narada environment (default: HF Space)
    BASE_MODEL     — base model ID (default: Qwen/Qwen3-1.7B)
    LORA_RANK      — LoRA rank (default: 16)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import nest_asyncio
nest_asyncio.apply()

import websockets
import torch
from datasets import Dataset
from unsloth import FastLanguageModel

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN     = os.environ["HF_TOKEN"]
HF_PUSH_REPO = os.environ["HF_PUSH_REPO"]
BASE_MODEL   = os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B")
ENV_URL      = os.environ.get("ENV_URL", "https://krishvenky-narada-env.hf.space")
LORA_RANK    = int(os.environ.get("LORA_RANK", "16"))
ADAPTER_NAME = "narada-detective-lora"

os.environ["HF_TOKEN"] = HF_TOKEN

CURRICULUM = [
    {"task": "monogenic",          "steps": 80},
    {"task": "oligogenic",         "steps": 60},
    {"task": "phenotype_mismatch", "steps": 60},
]
EVAL_SEEDS       = [42, 7, 999, 1337, 2024]
N_SEEDS_PER_TASK = 40
MAX_SEQ_LEN      = 2048
MINI_BATCH_SIZE  = 2
GRAD_ACCUM       = 4
LR               = 5e-6
WARMUP_STEPS     = 20

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert clinical geneticist. Generate a DIAGNOSTIC PLAN: 3-5 JSON action blocks
to navigate the gene-disease knowledge graph and identify the causal variant.

Output each action block on its own line. End with flag_causal when you are confident.

ACTIONS (one JSON per line, no other text):
  {"action_type": "hop",          "node_id": "<id>",       "reasoning": "<one sentence>"}
  {"action_type": "flag_causal",  "variant_id": "VAR:...", "reasoning": "<one sentence>"}
  {"action_type": "backtrack",                              "reasoning": "<one sentence>"}
  {"action_type": "summarise_trail",                        "reasoning": "<one sentence>"}

STRATEGY:
  1. Navigate phenotype -> disease -> gene -> variant chains.
  2. BRCA1/TP53 is a DECOY if phenotypes are cardiac/neurological -- skip it.
  3. Oligogenic: flag ALL causal variants, not just the first one.
  4. Flag before step 8 for a timing bonus.
  5. ABSENT PHENOTYPES are strong rule-out signals -- use them.
""").strip()

# ── Environment helpers ───────────────────────────────────────────────────────

def format_obs(obs: Dict[str, Any]) -> str:
    lines = [
        f"STEP {obs['step']}/{obs['max_steps']} | Task: {obs['task_type']}",
        "",
        "PATIENT PHENOTYPES (present):",
    ]
    for hid, name in zip(obs["patient_phenotypes"], obs["phenotype_names"]):
        lines.append(f"  + {hid} -- {name}")

    absent_ids   = obs.get("phenotypes_absent") or []
    absent_names = obs.get("phenotype_absent_names") or []
    if absent_ids:
        lines += ["", "ABSENT PHENOTYPES (rule-out signal):"]
        for hid, name in zip(absent_ids, absent_names):
            lines.append(f"  - {hid} -- {name}")

    n = obs["current_node"]
    lines += [
        "",
        f"CURRENT NODE: [{n['type'].upper()}] {n['name']} ({n['id']})",
        f"  Neighbors ({len(n['connected_node_ids'])}): {', '.join(n['connected_node_ids'][:8])}",
    ]
    if obs.get("trail"):
        trail = [f"{t['name']}({t['id']})" for t in obs["trail"][-4:]]
        lines.append(f"  Trail: {' -> '.join(trail)}")

    lines += ["", "CANDIDATE VARIANTS:"]
    for v in obs["candidate_variants"]:
        lines.append(
            f"  {v['id']} | {v['gene']} | {v['variant_type']} "
            f"| path={v['pathogenicity_score']:.2f} | {v['clinical_significance']}"
        )
    lines.append(f"\nStep reward: {obs['step_reward']:+.4f} | Cumulative: {obs['cumulative_reward']:.4f}")
    lines.append("Generate your diagnostic plan (3-5 JSON action blocks):")
    return "\n".join(lines)


def parse_all_actions(text: str) -> List[Dict[str, Any]]:
    actions = []
    for m in re.finditer(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL):
        try:
            d = json.loads(m.group(0))
            atype = str(d.get("action_type", "")).lower()
            if atype not in ("hop", "flag_causal", "backtrack", "summarise_trail", "request_lab"):
                continue
            actions.append({
                "action_type": atype,
                "node_id":     str(d["node_id"])    if d.get("node_id")    else None,
                "variant_id":  str(d["variant_id"]) if d.get("variant_id") else None,
                "reasoning":   str(d.get("reasoning", ""))[:200],
            })
            if atype == "flag_causal":
                break
        except Exception:
            continue
    return actions or [{"action_type": "summarise_trail", "reasoning": "fallback"}]


def parse_action(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"action_type": "summarise_trail", "reasoning": "fallback"}
    try:
        d = json.loads(m.group(0))
        atype = str(d.get("action_type", "summarise_trail")).lower()
        if atype not in ("hop", "flag_causal", "backtrack", "summarise_trail", "request_lab"):
            atype = "summarise_trail"
        return {
            "action_type": atype,
            "node_id":    str(d["node_id"])    if d.get("node_id")    else None,
            "variant_id": str(d["variant_id"]) if d.get("variant_id") else None,
            "reasoning":  str(d.get("reasoning", ""))[:200],
        }
    except Exception:
        return {"action_type": "summarise_trail", "reasoning": "parse error"}


async def run_episode_async(
    task_type: str,
    actions: List[Dict[str, Any]],
    seed: Optional[int] = None,
) -> float:
    ws_url = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    async with websockets.connect(ws_url, open_timeout=30, ping_interval=20) as ws:
        reset_msg: Dict[str, Any] = {"type": "reset", "task_type": task_type}
        if seed is not None:
            reset_msg["seed"] = seed
        await ws.send(json.dumps(reset_msg))
        raw = json.loads(await ws.recv())
        if raw.get("type") == "error":
            return 0.1

        obs = raw["data"]["observation"]
        last_reward = 0.1

        for action in actions:
            if obs.get("done"):
                break
            await ws.send(json.dumps({"type": "step", "action": action}))
            raw = json.loads(await ws.recv())
            if raw.get("type") == "error":
                break
            data = raw["data"]
            obs  = data["observation"]
            last_reward = data["reward"]
            if obs.get("done"):
                return float(last_reward)

        return float(last_reward)


async def collect_episode_async(task_type: str, seed: Optional[int] = None) -> List[Dict]:
    ws_url = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    steps: List[Dict] = []
    async with websockets.connect(ws_url, open_timeout=30, ping_interval=20) as ws:
        reset_msg: Dict[str, Any] = {"type": "reset", "task_type": task_type}
        if seed is not None:
            reset_msg["seed"] = seed
        await ws.send(json.dumps(reset_msg))
        raw = json.loads(await ws.recv())
        if raw.get("type") == "error":
            return steps
        obs = raw["data"]["observation"]
        steps.append({"prompt": format_obs(obs), "obs": obs, "task_type": task_type, "seed": seed})
    return steps


def run_episode(task_type: str, actions: List[Dict], seed: Optional[int] = None) -> float:
    return asyncio.get_event_loop().run_until_complete(run_episode_async(task_type, actions, seed))


def collect_episode(task_type: str, seed: Optional[int] = None) -> List[Dict]:
    return asyncio.get_event_loop().run_until_complete(collect_episode_async(task_type, seed))


# ── Reward function ───────────────────────────────────────────────────────────

def clamp(v: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, v)) if math.isfinite(v) else 0.1


async def _eval_one_async(text: str, task: str, seed: Any) -> float:
    actions = parse_all_actions(text)
    try:
        return clamp(await run_episode_async(task, actions, seed=seed))
    except Exception:
        return 0.1


def narada_reward(completions, prompts, task_type=None, seed=None, **kwargs):
    n     = len(completions)
    tasks = task_type if task_type is not None else ["monogenic"] * n
    seeds = seed       if seed      is not None else [None] * n

    texts = []
    for c in completions:
        if isinstance(c, list):
            texts.append(c[-1]["content"] if c else "")
        else:
            texts.append(str(c))

    async def _batch():
        return list(await asyncio.gather(*[
            _eval_one_async(t, task, s)
            for t, task, s in zip(texts, tasks, seeds)
        ]))

    return asyncio.get_event_loop().run_until_complete(_batch())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load model ────────────────────────────────────────────────────────────
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

    print(f"Loading {BASE_MODEL}...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = BASE_MODEL,
        max_seq_length= MAX_SEQ_LEN,
        dtype         = None,
        load_in_4bit  = True,
        token         = HF_TOKEN,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = LORA_RANK * 2,
        lora_dropout   = 0.0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    # Disable Qwen3 thinking mode
    _orig = tokenizer.apply_chat_template
    def _no_think(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig(*args, **kwargs)
    tokenizer.apply_chat_template = _no_think
    print("Thinking mode disabled.", flush=True)

    # ── Build dataset ─────────────────────────────────────────────────────────
    import random
    random.seed(42)
    train_seeds = random.sample(range(1, 10000), N_SEEDS_PER_TASK * 3)

    all_prompts = []
    for i, phase in enumerate(CURRICULUM):
        task  = phase["task"]
        seeds = train_seeds[i * N_SEEDS_PER_TASK : (i + 1) * N_SEEDS_PER_TASK]
        print(f"Collecting {N_SEEDS_PER_TASK} prompts for task={task}...", flush=True)
        for seed in seeds:
            steps = collect_episode(task, seed=seed)
            if steps:
                all_prompts.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": steps[0]["prompt"]},
                    ],
                    "task_type": task,
                    "seed":      seed,
                })

    dataset = Dataset.from_list(all_prompts)
    print(f"Dataset: {len(dataset)} prompts", flush=True)

    # ── GRPO config ───────────────────────────────────────────────────────────
    import shutil
    from trl import GRPOConfig, GRPOTrainer

    cache_path = "/tmp/unsloth_compiled_cache"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

    grpo_config = GRPOConfig(
        num_generations             = 8,
        temperature                 = 1.1,
        top_p                       = 0.95,
        learning_rate               = LR,
        per_device_train_batch_size = MINI_BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = WARMUP_STEPS,
        max_grad_norm               = 0.1,
        optim                       = "adamw_8bit",
        max_prompt_length           = 1200,
        max_completion_length       = 800,
        logging_steps               = 5,
        output_dir                  = f"/tmp/{ADAPTER_NAME}",
        report_to                   = "none",
    )

    # ── Curriculum training ───────────────────────────────────────────────────
    eval_results: Dict[str, float] = {}

    for phase in CURRICULUM:
        task    = phase["task"]
        n_steps = phase["steps"]

        phase_data = dataset.filter(lambda x: x["task_type"] == task)
        if len(phase_data) == 0:
            print(f"Skipping {task} — no data.", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Phase: {task}  |  {len(phase_data)} prompts  |  {n_steps} steps", flush=True)
        print(f"{'='*60}", flush=True)

        grpo_config.max_steps = n_steps

        trainer = GRPOTrainer(
            model            = model,
            processing_class = tokenizer,
            reward_funcs     = narada_reward,
            args             = grpo_config,
            train_dataset    = phase_data,
        )

        t0 = time.time()
        trainer.train()
        print(f"Phase {task} done in {(time.time()-t0)/60:.1f} min", flush=True)

        # Eval
        FastLanguageModel.for_inference(model)
        scores = []
        for es in EVAL_SEEDS:
            steps = collect_episode(task, seed=es)
            if not steps:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": steps[0]["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=200, temperature=0.3)
            completion = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
            scores.append(run_episode(task, [parse_action(completion)], seed=es))

        avg = sum(scores) / len(scores) if scores else 0.0
        eval_results[task] = avg
        print(f"Eval {task}: {avg:.4f}  (n={len(scores)})", flush=True)
        FastLanguageModel.for_training(model)

    # ── Save & push ───────────────────────────────────────────────────────────
    model.save_pretrained(ADAPTER_NAME)
    tokenizer.save_pretrained(ADAPTER_NAME)
    print(f"\nAdapter saved locally to ./{ADAPTER_NAME}", flush=True)

    model.push_to_hub(HF_PUSH_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_PUSH_REPO, token=HF_TOKEN)
    print(f"Pushed to https://huggingface.co/{HF_PUSH_REPO}", flush=True)

    print("\n=== TRAINING COMPLETE ===", flush=True)
    for task, score in eval_results.items():
        print(f"  {task:25s}: {score:.4f}", flush=True)


if __name__ == "__main__":
    main()
