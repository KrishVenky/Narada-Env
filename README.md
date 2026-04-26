---
title: Narada Env
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Narada — Rare Disease Diagnosis RL Environment

**Meta × PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale**

**Live environment:** [krishvenky-narada-env.hf.space](https://huggingface.co/spaces/KrishVenky/narada-env) · **Blog:** [Blog.md](https://huggingface.co/spaces/KrishVenky/narada-env/blob/main/Blog.md) · **Colab training notebook:** [Open in Colab](https://colab.research.google.com/drive/15tPrE95ASXcBA2zKImWgmRoFM0OozPQm?usp=sharing)

---

## Results

GRPO training on Qwen3-1.7B (LoRA rank 16, 200 steps curriculum) vs zero-shot baseline:

| Task | Baseline | After GRPO | Gain |
|---|---|---|---|
| monogenic | 0.4955 | **0.572** | +15.4% |
| oligogenic | 0.4955 | **0.561** | +13.2% |
| phenotype_mismatch | 0.4955 | **0.552** | +11.4% |
| **Average** | 0.4955 | **0.562** | **+13.3%** |

![Before/After GRPO](results/before_after.png)
*Zero-shot baseline vs. trained agent across all three task tiers*

![Reward Curve](results/reward_curve.png)
*Mean reward across 200 training steps (curriculum order). Shaded region = reward_std. Dotted line = zero-shot baseline.*

![Loss Curve](results/loss_curve.png)
*Policy loss across curriculum. Phase boundaries shown as dashed vertical lines.*

![Reward Std](results/reward_std.png)
*reward_std > 0 throughout confirms GRPO received real gradient signal — model never collapsed to uniform reward.*

---

## The Problem

Most rare disease patients wait **4–7 years** for a correct diagnosis ([Lancet 2024](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00056-1/fulltext), [Nature 2024](https://www.nature.com/articles/s41431-024-01604-z)). The delay is systemic, not a simple workflow gap:

- Rare diseases are **clinically heterogeneous** — many present with non-specific symptoms common across hundreds of conditions ([PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323401/))
- **60% of patients are initially misdiagnosed** with either a different physical illness or a psychological condition; patients consult 8+ specialists (EU) or 17+ specialists (US) before a correct diagnosis ([EURORDIS survey](https://www.eurordis.org/survey-reveals-lengthy-diagnostic-delays/))
- **"Medical ping-pong"** — patients are passed between specialists with poor cross-service communication ([NHS Genomics Education](https://www.genomicseducation.hee.nhs.uk/genotes/knowledge-hub/the-diagnostic-odyssey-in-rare-disease/))
- **95% of rare diseases have no approved treatment**, and poor clinician/patient awareness of symptom profiles compounds the problem ([Lancet 2024](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00056-1/fulltext))

ClinVar contains 2M+ catalogued genetic variants; HPO maps 15,000+ diseases to phenotypic signatures. The bottleneck is **reasoning under uncertainty** — cross-referencing patient symptoms against thousands of candidate variants while resisting high-salience but causally irrelevant signals.

This is exactly where current LLMs fail: they follow pathogenicity scores, not causal chains.

**Narada's scope:** The full diagnostic odyssey involves referral pathways, specialist availability, and fragmented care across institutions. Narada targets the **variant prioritization substage** — given a patient's phenotype profile and a shortlist of candidate genomic variants, which variant is causally responsible? This substage has a ground-truth signal (ClinVar), a tractable action space (graph traversal), and measurable improvement via RL — making it ideal for methodology proof-of-concept work.

---

## What Narada Is

A **reinforcement learning environment** where an LLM agent navigates a 55,000-node gene-disease knowledge graph built from real ClinVar and HPO data. The agent must diagnose a rare disease patient by reasoning through phenotype → disease → gene → variant chains.

**Three task tiers, increasing difficulty:**

| Task | Description | Key Challenge |
|---|---|---|
| `monogenic` | Single causal gene, 3–4 phenotypes | Basic directional reasoning |
| `oligogenic` | 2 causal genes (one variant each), 5–7 phenotypes | Multi-objective tracking across long trajectory |
| `phenotype_mismatch` | Cardiac patient + BRCA1 frameshift decoy | Causal discipline — resist the highest-salience wrong signal |

**Three-agent system:**
- **Detective (Qwen3-1.7B, trainable)** — navigates the graph, flags the causal variant
- **Overseer** — local heuristic (no LLM) that scores trajectory quality: penalises hallucinated hops, rewards touching the causal gene, and scales with a concise trail. Added only to *successful* terminal rewards.
- **Adversary** *(planned)* — curriculum case generation targeting Detective failure patterns; reliable adversarial curriculum from agent error logs is an open research problem, deferred to future work

---

## Proof-of-Concept Framing

This project uses Qwen3-1.7B as the Detective agent. At this scale, the honest goal is **methodology proof-of-concept**: can GRPO training on a verifiable graph-navigation task move the needle even on a constrained model? Measurable improvement from a low baseline is a legitimate research contribution.

The environment and training pipeline are designed to generalize to larger models — the same loop applies to Qwen2.5-72B with zero code changes.

---

## Graph

Built at runtime from:
- `data/hp.obo` — 19,389 HPO phenotype terms
- `data/clinvar_pathogenic.tsv` — 92,000 high-confidence pathogenic variants (GRCh38, criteria provided/expert panel, deduplicated)

**Graph stats:**
- 55,000+ nodes (phenotype, disease, gene, variant, pathway)
- 70,000+ edge pairs
- 3,268 genes represented

---

## Action Space

| Action | Effect | Reward |
|---|---|---|
| `hop(node_id)` | Move to connected node | +0.15 relevant / −0.05 irrelevant |
| `flag_causal(variant_id)` | Declare diagnosis; oligogenic cases allow multiple flags | +1.0 correct terminal, −0.5 wrong |
| `backtrack()` | Return to previous node | +0.05 after wrong direction |
| `request_lab(test)` | Get additional phenotype data | −0.10 penalty |
| `summarise_trail()` | Compressed visit summary | 0.0 (neutral) |

Signed raw rewards are mapped into OpenEnv's required score interval `(0.01, 0.99)`. This preserves penalties while keeping the validator-compatible output range.

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/KrishVenky/Narada-Env.git
cd Narada-Env

# 2. Create virtualenv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Generate filtered ClinVar (one-time, ~3 min)
python scripts/filter_clinvar.py

# 6. Start server
PYTHONPATH=src/envs uvicorn narada.server.app:app --port 7860

# 7. Open browser → http://localhost:7860
```

---

## Contributing

Pull requests welcome. To add a new task tier, reward component, or agent variant:

1. Fork the repo and create a branch off `main`
2. Environment logic lives in [src/envs/narada/](src/envs/narada/) — `environment.py` for reward design, `case_generator.py` for case sampling, `graph.py` for the knowledge graph
3. Add your changes and run the OpenEnv validator before opening a PR:
   ```bash
   openenv validate http://localhost:7860
   ```
4. Update `ARCHITECTURE.md` if you change reward values, node types, or the session protocol

**Key files:**

| File | Purpose |
|---|---|
| `src/envs/narada/server/environment.py` | Core RL loop, reward computation |
| `src/envs/narada/server/app.py` | FastAPI entry point (WebSocket + HTTP debug) |
| `src/envs/narada/graph.py` | Knowledge graph build + singleton |
| `src/envs/narada/case_generator.py` | Patient case generation per tier |
| `src/envs/narada/models.py` | Pydantic schemas (action/observation/state) |
| `src/envs/narada/client.py` | Python WebSocket client for agents |
| `training/narada_grpo.ipynb` | GRPO training notebook (Colab) |
| `inference.py` | Benchmark script (Groq or HF backend) |

---

## OpenEnv Validation

```bash
openenv validate https://krishvenky-narada-env.hf.space
```

All scores strictly in `(0.01, 0.99)`. `[END]` line always includes `score=` field.

---

## Baseline Benchmark

Zero-shot evaluation via `inference.py` (Groq backend, no fine-tuning). Two single-episode samples per model show the core problem: zero-shot LLMs are **inconsistent** and frequently collapse into `summarise_trail` loops instead of navigating the graph.

> Note: the exact scores below are reference runs from before reward hardening. Re-run `inference.py` before final submission so the README plots and tables match the current reward mapping.

### llama-3.3-70b-versatile (reference runs)

| Task | Run 1 Score | Run 2 Score | Notes |
|---|---|---|---|
| `monogenic` | **0.990** | 0.433 | Run 1: solved in 4 steps. Run 2: hit summarise_trail loop after step 3 |
| `oligogenic` | 0.500 | 0.240 | Run 1: WS disconnect mid-episode. Run 2: full summarise_trail timeout |
| `phenotype_mismatch` | 0.060 | 0.225 | Run 1: looped on wrong gene 9× before giving up. Run 2: pure timeout |

### Multi-model comparison (zero-shot, all 3 tasks, single run each)

| Model | monogenic | oligogenic | phenotype_mismatch | Behavior |
|---|---|---|---|---|
| `llama-3.3-70b-versatile` | **0.990** | 0.500 | 0.060 | Hops graph; inconsistent |
| `llama-3.1-8b-instant` | 0.310 | 0.425 | 0.310 | Hops but flags wrong variant |
| `mixtral-8x7b-32768` | 0.233 | 0.220 | 0.225 | Full summarise_trail timeout |
| `gemma2-9b-it` | 0.233 | 0.220 | 0.225 | Full summarise_trail timeout |

The pattern is clear: large frontier models (llama-3.3-70b) occasionally navigate the graph correctly but are inconsistent (0.990 vs 0.433 on the same task across runs). Mid-size models (llama-3.1-8b) attempt navigation but misfire on the final flag. Smaller models (mixtral, gemma2) collapse entirely to the `summarise_trail` loop and never issue a `flag_causal`. Fine-tuning on the graph-navigation reward signal is intended to make correct phenotype → gene → variant chaining the default, not a lucky outcome that only large models achieve occasionally.

**Target post-GRPO (Qwen3-1.7B):** consistent flag accuracy > 50% on monogenic, causal path coverage > 60%.

> To switch backends: set `GROQ_API_KEY` for Groq, or `HF_TOKEN` for HF Inference Router. The script auto-detects which to use.

---

## Training

Training notebook: `training/narada_grpo.ipynb` (Colab, Unsloth + HF TRL GRPO)

Base model: `Qwen/Qwen3-1.7B` — fits on a free T4 (4-bit quantised, 17.4M trainable LoRA params)

Tasks trained in curriculum order: `monogenic → oligogenic → phenotype_mismatch`

### Architecture

**Multi-step outcome GRPO** — the model generates a complete 3–5 action *diagnostic plan* per prompt. The full plan is executed in the environment and the terminal reward (correct flag / wrong flag / timeout) becomes the training signal. This gives 10× more reward variance than single-step GRPO:

| Approach | Typical reward range | reward_std | Learning |
|---|---|---|---|
| 1-step (old) | 0.47–0.56 | ~0.03 | Slow |
| Multi-step plan | 0.28–0.99 | ~0.20–0.35 | Strong |

**Async-parallel reward** — all G=8 completions are evaluated concurrently via `asyncio.gather`. Each completion's plan runs as an independent WebSocket session. Total overhead ≈ one episode round-trip, not 8×.

**Curriculum learning** — monogenic cases establish the basic hop→flag behaviour before oligogenic (multi-objective) and phenotype_mismatch (decoy resistance) add harder constraints.

**Milestone reward** — environment gives a +0.10 bonus the first time the agent visits the actual causal gene node. Creates a two-stage reward landscape: find the gene (+0.10) → flag the variant (+1.0).

### Key training config

| Param | Value | Why |
|---|---|---|
| `num_generations` | 8 | More completions per prompt → higher reward_std |
| `temperature` | 1.1 | Forces diverse hop targets within a group |
| `max_completion_length` | 800 | Fits 4–5 JSON action blocks |
| `N_SEEDS_PER_TASK` | 40 | 120 total prompts (was 60) — more case diversity |
| `LR` | 5e-6 | Conservative to avoid catastrophic forgetting on 1.7B |

---

## Multi-Model Benchmark

Run zero-shot comparison across sub-5B models (all runnable on T4):

```bash
HF_TOKEN=hf_... python benchmark.py --n_seeds 3
```

Models compared: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B, Phi-3.5-mini

Results are saved to `benchmark_results.json` after each run.

---

## Graph Export (Neo4j)

Export the 55K-node knowledge graph to Cypher for Neo4j Aura (free tier) or Desktop:

```bash
PYTHONPATH=src/envs python scripts/export_neo4j.py
# Generates neo4j_nodes.cypher + neo4j_rels.cypher
# Import into Neo4j Browser, then:
# MATCH (n:NaradaNode) RETURN n LIMIT 100
```

The live environment also exposes a subgraph JSON endpoint for D3.js visualization:
```
GET /graph/subgraph?node_id=GENE:BRCA2&depth=2
```
