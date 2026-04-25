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

**Live environment:** [krishvenky-narada-env.hf.space](https://krishvenky-narada-env.hf.space)

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
| `oligogenic` | 2–3 causal genes, 5–7 phenotypes | Multi-objective tracking across long trajectory |
| `phenotype_mismatch` | Cardiac patient + BRCA1 frameshift decoy | Causal discipline — resist the highest-salience wrong signal |

**Three-agent system:**
- **Detective (Qwen3-1.7B, trainable)** — navigates the graph, flags the causal variant
- **Overseer** — evaluates reasoning quality, penalises hallucinated hops
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
| `flag_causal(variant_id)` | Declare diagnosis (terminal) | +1.0 correct, −0.5 wrong |
| `backtrack()` | Return to previous node | +0.05 after wrong direction |
| `request_lab(test)` | Get additional phenotype data | −0.10 penalty |
| `summarise_trail()` | Compressed visit summary | 0.0 (neutral) |

All scores are clamped to `(0.01, 0.99)`.

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
| `src/envs/narada/environment.py` | Core RL loop, reward computation |
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

### llama-3.3-70b-versatile (reference runs)

| Task | Run 1 Score | Run 2 Score | Notes |
|---|---|---|---|
| `monogenic` | **0.990** | 0.433 | Run 1: solved in 4 steps. Run 2: hit summarise_trail loop after step 3 |
| `oligogenic` | 0.500 | 0.240 | Run 1: WS disconnect mid-episode. Run 2: full summarise_trail timeout |
| `phenotype_mismatch` | 0.060 | 0.225 | Run 1: looped on wrong gene 9× before giving up. Run 2: pure timeout |

### Multi-model comparison (zero-shot, monogenic task, seed=42)

| Model | Score | Steps | Behavior |
|---|---|---|---|
| `llama-3.3-70b-versatile` | 0.990 | 4 | Correct flag in 4 hops |
| `llama-3.1-8b-instant` | 0.433 | 15 | summarise_trail loop |
| `mixtral-8x7b-32768` | 0.433 | 15 | summarise_trail loop |
| `gemma2-9b-it` | 0.433 | 15 | summarise_trail loop |

The variance across model sizes (0.990 vs 0.433) is the core motivation for GRPO training: larger models occasionally find the right path but cannot do so reliably, and smaller models collapse to the `summarise_trail` fallback immediately. Fine-tuning on the graph-navigation reward signal is intended to make correct phenotype → gene → variant chaining the default behavior, not the lucky outcome.

**Target post-GRPO (Qwen3-1.7B):** consistent flag accuracy > 50% on monogenic, causal path coverage > 60%.

> To switch backends: set `GROQ_API_KEY` for Groq, or `HF_TOKEN` for HF Inference Router. The script auto-detects which to use.

---

## Training

Training notebook: `training/narada_grpo.ipynb` (Colab, Unsloth + HF TRL GRPO)

Base model: `Qwen/Qwen3-1.7B`

Tasks trained in curriculum order: monogenic → oligogenic → phenotype_mismatch
