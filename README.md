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

---

## The Problem

Most rare disease patients wait **4–7 years** for a correct diagnosis ([Lancet 2024](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00056-1/fulltext), [Nature 2024](https://www.nature.com/articles/s41431-024-01604-z)). The delay is systemic, not a simple workflow gap:

- Rare diseases are **clinically heterogeneous** — many present with non-specific symptoms common across hundreds of conditions ([PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323401/))
- **60% of patients are initially misdiagnosed** with either a different physical illness or a psychological condition; patients consult 8+ specialists (EU) or 17+ specialists (US) before a correct diagnosis ([EURORDIS survey](https://www.eurordis.org/survey-reveals-lengthy-diagnostic-delays/))
- **"Medical ping-pong"** — patients are passed between specialists with poor cross-service communication ([NHS Genomics Education](https://www.genomicseducation.hee.nhs.uk/genotes/knowledge-hub/the-diagnostic-odyssey-in-rare-disease/))
- **95% of rare diseases have no approved treatment**, and poor clinician/patient awareness of symptom profiles compounds the problem ([Lancet 2024](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00056-1/fulltext))

ClinVar contains 2M+ catalogued genetic variants; HPO maps 15,000+ diseases to phenotypic signatures. The bottleneck is **reasoning under uncertainty** — cross-referencing patient symptoms against thousands of candidate variants while resisting high-salience but causally irrelevant signals.

This is exactly where current LLMs fail: they follow pathogenicity scores, not causal chains.

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
- **Adversary (exploratory)** — attempts to generate harder cases targeting Detective failure patterns *(work in progress — reliable curriculum generation from agent error logs is an open research problem)*

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
# 1. Generate filtered ClinVar (one-time, ~3 min)
python scripts/filter_clinvar.py

# 2. Install
pip install -r requirements.txt

# 3. Start server
PYTHONPATH=src/envs uvicorn narada.server.app:app --port 7860

# 4. Open browser → http://localhost:7860
```

---

## OpenEnv Validation

```bash
openenv validate https://krishvenky-narada-env.hf.space
```

All scores strictly in `(0.01, 0.99)`. `[END]` line always includes `score=` field.

---

## Training

Training notebook: `training/narada_grpo.ipynb` (Colab, Unsloth + HF TRL GRPO)

Base model: `Qwen/Qwen3-1.7B`

Tasks trained in curriculum order: monogenic → oligogenic → phenotype_mismatch
