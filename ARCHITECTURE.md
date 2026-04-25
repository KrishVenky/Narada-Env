# Narada — Architecture & Reward Design

## Overview

Narada is a multi-step reinforcement learning environment where an LLM agent plays a clinical geneticist navigating a gene-disease knowledge graph. The agent observes a patient's HPO phenotype terms and a pool of candidate variants, then navigates the graph to identify the causal variant(s).

---

## Knowledge Graph

Built at runtime from two public biomedical datasets:

| Source | File | Content |
|--------|------|---------|
| HPO | `data/hp.obo` | 19,389 phenotype terms + hierarchy |
| ClinVar | `data/clinvar_pathogenic.tsv` | 92,000 high-confidence pathogenic variants (GRCh38, criteria provided / expert panel, deduplicated by AlleleID) |

### Node types

| Type | ID format | Count | Meaning |
|------|-----------|-------|---------|
| `phenotype` | `HP:XXXXXXX` | ~2,400 | HPO terms (catalog + 5-level ancestors) |
| `disease` | `DIS:<slug>` | ~28,000 | Disease names from ClinVar PhenotypeList |
| `gene` | `GENE:<symbol>` | ~3,268 | Gene symbols |
| `variant` | `VAR:<AlleleID>` | ~43,634 | Individual ClinVar variants |
| `pathway` | `PATH:<name>` | ~14 | Coarse pathway group (cardiac, neurological, …) |

**Total: 55,201 nodes, 70,741 edge-pairs**

### Edge structure

All edges are undirected (bidirectional). Key connections:

```
phenotype ↔ phenotype   (HPO parent-child hierarchy)
phenotype ↔ disease     (catalog wiring + inverted word-index matching)
disease   ↔ gene        (via ClinVar PhenotypeList)
gene      ↔ variant     (one gene → many variants)
gene      ↔ pathway     (PATHWAY_MAP classification)
```

### Graph build performance

The `_add_hpo_nodes()` method uses an O(N+M) inverted word index instead of O(N×M) brute-force matching. Cold build time: ~2 seconds. Loaded once at server startup, shared across all sessions.

---

## Session Lifecycle

```
Client                          Server
  │                                │
  ├─── WebSocket connect ─────────►│  NaradaEnvironment() created (per session)
  │                                │
  ├─── {"type":"reset", ...} ─────►│  generate_case() → PatientCase
  │◄── {"type":"observation"} ─────┤  _build_observation() → StepResult(reward=0.0)
  │                                │
  ├─── {"type":"step", action} ───►│  _dispatch_action() → raw reward
  │◄── {"type":"observation"} ─────┤  reward mapped to (0.01, 0.99)
  │         (repeat)               │
  │                                │
  ├─── flag_causal(VAR:xxxxx) ────►│  _compute_terminal_reward() + _overseer_score()
  │◄── {"type":"observation",      │  done=True, final reward returned
  │     done:true, reward:X} ──────┤
  │                                │
  └─── WebSocket close ───────────►│  session destroyed
```

**WORKERS=1** is enforced. All state is in-memory per WebSocket connection; no shared state between sessions.

---

## Task Tiers

### monogenic (easy)
- 1 causal gene, 3–4 HPO phenotypes, 5–8 candidate variants
- Max 15 steps
- Tests: basic phenotype → disease → gene → variant chain reasoning

### oligogenic (medium)
- 2 causal genes, 5–7 HPO phenotypes, 10–15 candidates
- Max 25 steps
- Tests: multi-objective tracking — the agent must flag both contributing variants; correct intermediate flags are recorded and the episode continues until all are found or a wrong variant is flagged

### phenotype_mismatch (hard)
- 1 causal gene (cardiac/neurological)
- A high-pathogenicity BRCA1/BRCA2/TP53 frameshift variant in the candidate pool as a decoy
- Max 20 steps
- Tests: causal discipline — pathogenicity score alone is not sufficient; the variant must match the patient's phenotype

---

## Reward Design

### Step-level rewards

| Action | Reward | Condition |
|--------|--------|-----------|
| `hop` | +0.15 | Target node is on the causal path |
| `hop` | −0.05 | Target node is off-path |
| `hop` | −0.10 | Hallucinated hop (node exists but not connected) |
| `backtrack` | +0.05 | Previous node was off-path (recovering) |
| `backtrack` | −0.05 | Previous node was on-path (wrong direction) |
| `request_lab` | −0.10 | Always penalised |
| `summarise_trail` | 0.00 | Neutral |
| Per-step | −0.01 | Applied to every action (efficiency pressure) |

### Terminal rewards

| Outcome | Reward |
|---------|--------|
| Correct flag (monogenic/mismatch) | +1.0 |
| Correct flag + timing bonus (step < 10) | +1.2 |
| Progress per correct variant (oligogenic) | `0.5 / total` non-terminal |
| All oligogenic variants flagged | `(correct/total) × 0.5` + timing bonus |
| Flagged decoy in mismatch task | −0.5 |
| Wrong flag | −0.5 |
| Timeout (no flag) | `-0.25 + min(0.2, trail_size / max_steps × 0.25)` |

### Overseer score (additive, 0.0–0.3)

Added only to successful terminal rewards. Computed locally without an LLM call:

| Criterion | Effect |
|-----------|--------|
| Hallucinated hops | −0.05 each |
| Visited < 3 unique nodes | −0.10 |
| Visited causal gene node | +0.05 |

### Score mapping

Rewards are kept as signed raw values internally, then mapped to the **open interval (0.01, 0.99)** before returning to the client. This preserves the ordering between penalties, neutral moves, and successes while satisfying the OpenEnv validator. `math.isfinite()` guards against NaN/inf.

---

## Three-Agent System

### Detective (trainable)
- Qwen3-1.7B fine-tuned with GRPO
- Navigates the graph, flags the causal variant
- Trained via `training/narada_grpo.ipynb`

### Overseer (local)
- Heuristic, no LLM call: reads only the trail and hop counters, not the free-form `reasoning` string
- Penalises hallucinated hops and trivial exploration
- Adds 0.0–0.3 only to successful terminal rewards

### Adversary (exploratory, WIP)
- Intended to generate harder cases targeting Detective failure patterns
- Reliable curriculum generation from agent error logs is an open research problem
- Current implementation falls back to random seed selection

---

## Action Space

```python
class NaradaAction(BaseModel):
    action_type: str   # hop | flag_causal | backtrack | request_lab | summarise_trail
    node_id:     Optional[str]  # required for hop
    variant_id:  Optional[str]  # required for flag_causal (format: VAR:12345)
    test_type:   Optional[str]  # for request_lab
    reasoning:   str            # agent's stated rationale (logged by Overseer)
```

## Observation Space

```python
class NaradaObservation(BaseModel):
    step:               int
    max_steps:          int
    task_type:          str
    current_node:       GraphNode
    trail:              List[GraphNode]   # last 10 visited nodes
    patient_phenotypes: List[str]         # HPO IDs
    phenotype_names:    List[str]
    candidate_variants: List[Variant]     # 5–15 variants
    step_reward:        float
    cumulative_reward:  float
    done:               bool
    info:               Dict[str, Any]    # ground_truth_hint revealed on done=True
```

---

## OpenEnv Compliance

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check |
| `/metadata` | GET | Environment name + description |
| `/schema` | GET | Pydantic JSON schemas for action/observation/state |
| `/mcp` | POST | JSON-RPC 2.0 tool discovery |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action |
| `/state` | GET | Current episode metadata |
| `/ws` | WebSocket | Primary transport (persistent session) |

Validation: `python -m openenv.cli validate --url https://krishvenky-narada-env.hf.space` → **6/6 passed**
