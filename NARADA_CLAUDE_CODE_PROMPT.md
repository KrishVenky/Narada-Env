# NARADA-ENV: Claude Code Agent Prompt
# Paste this entire file into Claude Code as your starting prompt.
# Do NOT run steps out of order — each phase depends on the previous.

---

## CONTEXT: What This Codebase Is

You are working on **Narada-Env**, a reinforcement learning environment built on
Meta's OpenEnv framework for training AI agents to do **clinical genomic variant
prioritisation** — specifically, given a patient's HPO phenotypes and a set of
candidate gene variants from an exome panel, navigate a biomedical knowledge graph
to identify the causal variant while resisting high-salience distractors.

The existing repo is at: https://github.com/KrishVenky/Narada-Env

The graph contains:
- 28,000 disease nodes
- 3,268 gene nodes  
- 92,000 pathogenic variant nodes
- HPO phenotype ontology edges
- Gene-disease, variant-gene, phenotype-disease associations

**Your job is to implement 5 specific upgrades across this session, in order.**
All persistent data must live in Neo4j AuraDB (cloud). No local SQLite, no pickle
files, no JSON dumps as the source of truth. Local files are only for code.

---

## ENVIRONMENT SETUP — Do This First

### Step 1: Install all dependencies

```bash
pip install neo4j python-dotenv groq anthropic openai requests tqdm rich pandas
pip install openenv-sdk   # or whatever the local install path is
```

### Step 2: Create `.env` file (do NOT hardcode credentials anywhere)

```
NEO4J_URI=neo4j+s://<your-aura-instance>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-aura-password>
GROQ_API_KEY=<your-groq-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
```

### Step 3: Verify Neo4j AuraDB connection

Write a file `db/connection.py`:

```python
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
    )

def verify_connection():
    with get_driver().session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS total LIMIT 1")
        print(f"Connected. Total nodes: {result.single()['total']}")

if __name__ == "__main__":
    verify_connection()
```

Run `python db/connection.py` and confirm it connects before proceeding.

---

## PHASE 1: Neo4j Schema + Data Migration (~45 min)

### Goal
Move the graph data (currently loaded from local files) into Neo4j AuraDB so all
queries are cloud-native. The environment should never read a local CSV/JSON for
graph traversal — it should query Neo4j every time.

### Task 1.1 — Write `db/schema.py`

Create constraints and indexes:

```cypher
-- Run these via session.run() in schema.py

CREATE CONSTRAINT gene_id_unique IF NOT EXISTS
FOR (g:Gene) REQUIRE g.gene_id IS UNIQUE;

CREATE CONSTRAINT disease_id_unique IF NOT EXISTS  
FOR (d:Disease) REQUIRE d.disease_id IS UNIQUE;

CREATE CONSTRAINT variant_id_unique IF NOT EXISTS
FOR (v:Variant) REQUIRE v.variant_id IS UNIQUE;

CREATE CONSTRAINT hpo_id_unique IF NOT EXISTS
FOR (h:HPO) REQUIRE h.hpo_id IS UNIQUE;

CREATE INDEX variant_cadd IF NOT EXISTS
FOR (v:Variant) ON (v.cadd_score);

CREATE INDEX gene_name IF NOT EXISTS
FOR (g:Gene) ON (g.symbol);
```

### Task 1.2 — Write `db/ingest.py`

Read whatever local source format exists (CSV, JSON, pickle — check the repo).
Batch-insert into Neo4j using `UNWIND` for performance. Use batches of 500 nodes.

Critical relationships to create:
```cypher
(:Gene)-[:HAS_VARIANT]->(:Variant)
(:Variant)-[:ASSOCIATED_WITH_DISEASE]->(:Disease)
(:Disease)-[:HAS_PHENOTYPE]->(:HPO)
(:Gene)-[:EXPRESSED_IN_DISEASE]->(:Disease)
(:HPO)-[:IS_A]->(:HPO)           # ontology hierarchy
```

Each Variant node must store:
- `variant_id`, `cadd_score`, `gene_id`, `pathogenicity_class` (P/LP/VUS)
- `associated_hpo_ids` (list) — phenotypes this variant is reported with

Each Gene node must store:
- `gene_id`, `symbol`, `omim_id`
- `typical_phenotype_ids` (list) — HPO terms strongly associated with this gene

### Task 1.3 — Verify data is queryable

Write a quick test in `db/verify.py`:
```python
# Should return a non-empty result:
# MATCH (g:Gene)-[:HAS_VARIANT]->(v:Variant) 
# WHERE v.cadd_score > 20 RETURN g.symbol, v.cadd_score LIMIT 10
```

---

## PHASE 2: Upgrade the Observation Space — Add Negative Phenotypes (~30 min)

### Goal
The current observation gives the agent `phenotypes_present`. Add `phenotypes_absent`
— HPO terms that are **strongly associated with candidate genes but NOT present in
the patient**. This enables exclusion reasoning, which graph search cannot do.

### Task 2.1 — Write `environment/observation_builder.py`

```python
from db.connection import get_driver

def build_observation(case: dict, current_gene: str) -> dict:
    """
    Constructs the full observation for a single environment step.
    
    Args:
        case: dict with patient_hpo_ids, causal_gene, candidate_genes
        current_gene: the gene node the agent is currently at
    
    Returns:
        observation dict with present AND absent phenotypes
    """
    driver = get_driver()
    
    with driver.session() as session:
        # Get phenotypes strongly associated with candidate genes
        # that the patient does NOT have
        result = session.run("""
            UNWIND $candidate_genes AS gene_sym
            MATCH (g:Gene {symbol: gene_sym})-[:EXPRESSED_IN_DISEASE]->(d:Disease)
                  -[:HAS_PHENOTYPE]->(h:HPO)
            WHERE NOT h.hpo_id IN $patient_hpo_ids
            WITH h.hpo_id AS hpo_id, count(*) AS frequency
            WHERE frequency >= 3
            RETURN hpo_id
            ORDER BY frequency DESC
            LIMIT 10
        """, 
        candidate_genes=case['candidate_genes'],
        patient_hpo_ids=case['patient_hpo_ids'])
        
        absent_phenotypes = [r['hpo_id'] for r in result]
        
        # Get neighbors of current node for traversal options
        neighbors = session.run("""
            MATCH (g:Gene {symbol: $gene})-[:HAS_VARIANT]->(v:Variant)
            RETURN v.variant_id, v.cadd_score, v.pathogenicity_class,
                   v.associated_hpo_ids
            ORDER BY v.cadd_score DESC
        """, gene=current_gene)
        
        candidate_variants = [dict(r) for r in neighbors]
    
    return {
        "current_gene": current_gene,
        "phenotypes_present": case['patient_hpo_ids'],
        "phenotypes_absent": absent_phenotypes,          # ← KEY ADDITION
        "candidate_variants": candidate_variants,
        "step": case.get('current_step', 0),
        "max_steps": case.get('max_steps', 10),
    }
```

### Task 2.2 — Add exclusion reasoning reward in `environment/reward.py`

Add this function and call it from your main reward calculation:

```python
def score_exclusion_reasoning(reasoning: str, phenotypes_absent: list) -> float:
    """
    Bonus reward when the agent explicitly references absent phenotypes.
    This signals the agent is doing exclusion logic, not just graph traversal.
    """
    if not reasoning or not phenotypes_absent:
        return 0.0
    
    mentioned = sum(1 for hpo in phenotypes_absent if hpo in reasoning)
    # Small bonus, capped — we don't want agents gaming this
    return round(0.08 * min(mentioned, 2), 3)
```

---

## PHASE 3: Differential Hypothesis Tracking (~45 min)

### Goal
Force the agent to maintain a ranked differential diagnosis across steps, not just
hop to one answer. This is the clearest structural difference from graph search.

### Task 3.1 — Update the action schema in `environment/actions.py`

The agent's action must now include a `differential` field:

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NaradaAction:
    hop_to: str                        # gene/variant to move to
    reasoning: str                     # free-text justification
    differential: List[dict]           # ← NEW: ranked hypothesis list
    
    # differential format:
    # [
    #   {"gene": "BRCA2", "confidence": 0.7, "evidence": "3/4 phenotypes match, low absent overlap"},
    #   {"gene": "PALB2", "confidence": 0.2, "evidence": "high CADD but absent HP:0000822"},
    #   {"gene": "RAD51", "confidence": 0.1, "evidence": "partial phenotype, weak association"},
    # ]
    
    def validate(self):
        assert 1 <= len(self.differential) <= 5, "Differential must have 1-5 entries"
        assert all('gene' in d and 'confidence' in d for d in self.differential)
        assert abs(sum(d['confidence'] for d in self.differential) - 1.0) < 0.01, \
               "Confidences must sum to 1.0"
```

### Task 3.2 — Score differential quality in Overseer

Add `score_differential()` to your Overseer scoring logic:

```python
def score_differential(action: NaradaAction, causal_gene: str) -> float:
    """
    Scores how well the agent's differential tracks the true answer.
    
    Returns float in [-0.15, +0.2]
    """
    score = 0.0
    genes_ranked = [d['gene'] for d in action.differential]
    
    if not genes_ranked:
        return -0.1
    
    # Strong bonus: causal gene is top hypothesis
    if genes_ranked[0] == causal_gene:
        score += 0.2
    # Partial bonus: causal gene is in differential at all
    elif causal_gene in genes_ranked:
        rank = genes_ranked.index(causal_gene)
        score += 0.1 / rank  # decays with rank
    else:
        # Causal gene not even considered — significant penalty
        score -= 0.1
    
    # Penalty: if a phenotype_mismatch decoy is ranked #1 with high confidence
    top_hypothesis = action.differential[0]
    if top_hypothesis['gene'] != causal_gene and top_hypothesis['confidence'] > 0.6:
        score -= 0.15  # agent anchored on distractor
    
    return round(score, 3)
```

### Task 3.3 — Store differential history in Neo4j

Each episode's differential trajectory should be persisted so you can analyze
how the agent's hypothesis evolves over steps:

```python
def log_episode_step(episode_id: str, step: int, action: NaradaAction, reward: float):
    with get_driver().session() as session:
        session.run("""
            MERGE (e:Episode {episode_id: $episode_id})
            CREATE (s:Step {
                step_number: $step,
                hop_to: $hop_to,
                reasoning: $reasoning,
                reward: $reward,
                top_hypothesis: $top_hypothesis,
                differential_size: $diff_size,
                timestamp: datetime()
            })
            MERGE (e)-[:HAS_STEP]->(s)
        """,
        episode_id=episode_id,
        step=step,
        hop_to=action.hop_to,
        reasoning=action.reasoning,
        reward=reward,
        top_hypothesis=action.differential[0]['gene'] if action.differential else None,
        diff_size=len(action.differential))
```

---

## PHASE 4: Failure-Mode Adversary — Real Implementation (~90 min)

### Goal
Replace the random-seed stub with a real adversary that generates cases by
amplifying the distractor strength in documented failure modes. Cases must be
guaranteed solvable (true path untouched) but harder (stronger distractor).

### Task 4.1 — Write `adversary/failure_analyzer.py`

This reads episode logs from Neo4j and categorises failure modes:

```python
from db.connection import get_driver
from collections import defaultdict

class FailureAnalyzer:
    """
    Reads episode history from Neo4j and identifies patterns in agent failures.
    """
    
    def __init__(self):
        self.driver = get_driver()
    
    def get_failure_modes(self, min_failures: int = 3) -> dict:
        """
        Returns failure cases grouped by type.
        Requires at least min_failures examples to count as a pattern.
        """
        with self.driver.session() as session:
            # Cases where agent chose high-CADD decoy over true gene
            cadd_anchored = session.run("""
                MATCH (e:Episode)-[:HAS_STEP]->(s:Step)
                WHERE e.outcome = 'failure'
                  AND s.step_number = e.final_step
                MATCH (v:Variant {variant_id: s.hop_to})
                WHERE v.cadd_score > 25 AND v.gene_id <> e.causal_gene
                RETURN e.episode_id, e.causal_gene, e.case_id,
                       v.cadd_score as decoy_cadd, s.top_hypothesis as chosen
                ORDER BY v.cadd_score DESC
            """).data()
            
            # Cases where agent got lost (too many hops)
            navigation_lost = session.run("""
                MATCH (e:Episode)
                WHERE e.outcome = 'failure' AND e.total_steps > 8
                RETURN e.episode_id, e.causal_gene, e.case_id, e.total_steps
            """).data()
            
            # Cases where phenotype overlap confused agent
            phenotype_confused = session.run("""
                MATCH (e:Episode)-[:HAS_STEP]->(s:Step)
                WHERE e.outcome = 'failure'
                  AND s.step_number = e.final_step
                  AND s.top_hypothesis <> e.causal_gene
                MATCH (g:Gene {symbol: s.top_hypothesis})
                      -[:EXPRESSED_IN_DISEASE]->(d:Disease)-[:HAS_PHENOTYPE]->(h:HPO)
                WHERE h.hpo_id IN e.patient_hpo_ids
                WITH e, s, count(h) as overlap_count
                WHERE overlap_count >= 3
                RETURN e.episode_id, e.causal_gene, e.case_id, 
                       s.top_hypothesis as confused_with, overlap_count
            """).data()
        
        return {
            'cadd_anchor': cadd_anchored if len(cadd_anchored) >= min_failures else [],
            'navigation_lost': navigation_lost if len(navigation_lost) >= min_failures else [],
            'phenotype_confused': phenotype_confused if len(phenotype_confused) >= min_failures else [],
        }
```

### Task 4.2 — Write `adversary/case_generator.py`

```python
from db.connection import get_driver
from adversary.failure_analyzer import FailureAnalyzer
import random
import copy
import uuid

class FailureModeAdversary:
    """
    Generates new training cases by amplifying distractors in known failure modes.
    
    Key property: ALL generated cases are solvable because the true causal gene
    and its path are preserved. Only the distractor gene/variant is swapped.
    
    This satisfies the PAIRED requirement: challenging but solvable.
    """
    
    def __init__(self):
        self.driver = get_driver()
        self.analyzer = FailureAnalyzer()
        self.failure_modes = {}
        self.refresh_failure_modes()
    
    def refresh_failure_modes(self):
        """Call this periodically to update from latest episode logs."""
        self.failure_modes = self.analyzer.get_failure_modes()
        print(f"Loaded failure modes: "
              f"{len(self.failure_modes['cadd_anchor'])} CADD-anchor, "
              f"{len(self.failure_modes['phenotype_confused'])} phenotype-confused")
    
    def generate_case(self, difficulty: str = 'auto') -> dict:
        """
        difficulty: 'cadd_anchor' | 'phenotype_confused' | 'navigation' | 'auto'
        'auto' picks the most populated failure mode.
        """
        if difficulty == 'auto':
            # Pick failure mode with most examples
            difficulty = max(
                self.failure_modes,
                key=lambda k: len(self.failure_modes[k])
            )
            if not self.failure_modes[difficulty]:
                # No failures logged yet — fall back to random (early training)
                print("No failure modes yet, using random seed")
                return self._random_case()
        
        if difficulty == 'cadd_anchor':
            return self._amplify_cadd_distractor()
        elif difficulty == 'phenotype_confused':
            return self._amplify_phenotype_confusion()
        else:
            return self._random_case()
    
    def _amplify_cadd_distractor(self) -> dict:
        """
        Takes a case where agent anchored on high CADD.
        Finds an EVEN HIGHER CADD variant from a different gene to use as decoy.
        True causal gene and path are untouched.
        """
        base_failures = self.failure_modes['cadd_anchor']
        if not base_failures:
            return self._random_case()
        
        base = random.choice(base_failures)
        
        with self.driver.session() as session:
            # Load original case
            original = session.run("""
                MATCH (c:Case {case_id: $case_id})
                RETURN c
            """, case_id=base['case_id']).single()
            
            if not original:
                return self._random_case()
            
            case_data = dict(original['c'])
            
            # Find a stronger CADD decoy: higher score, different gene, overlapping phenotypes
            stronger_decoy = session.run("""
                MATCH (v:Variant)-[:ASSOCIATED_WITH_DISEASE]->(d:Disease)
                      -[:HAS_PHENOTYPE]->(h:HPO)
                WHERE v.cadd_score > $threshold
                  AND v.gene_id <> $true_gene
                  AND h.hpo_id IN $patient_hpos
                WITH v, count(h) as phenotype_hits
                WHERE phenotype_hits >= 1
                RETURN v.variant_id, v.cadd_score, v.gene_id, phenotype_hits
                ORDER BY v.cadd_score DESC
                LIMIT 1
            """,
            threshold=base.get('decoy_cadd', 20),
            true_gene=base['causal_gene'],
            patient_hpos=case_data.get('patient_hpo_ids', [])).single()
            
            if not stronger_decoy:
                return self._random_case()
            
            # Build new case with amplified distractor
            new_case = copy.deepcopy(case_data)
            new_case['case_id'] = f"adv_{uuid.uuid4().hex[:8]}"
            new_case['decoy_variant_id'] = stronger_decoy['variant_id']
            new_case['decoy_gene'] = stronger_decoy['gene_id']
            new_case['source'] = 'adversary_cadd_amplified'
            new_case['difficulty_target'] = 'cadd_anchor'
            new_case['guaranteed_solvable'] = True  # true path untouched
            
            # Persist generated case back to Neo4j
            self._persist_generated_case(new_case)
            
            return new_case
    
    def _amplify_phenotype_confusion(self) -> dict:
        """
        Takes a case where agent was confused by phenotype overlap.
        Finds a gene with EVEN MORE phenotype overlap to the patient as decoy.
        """
        base_failures = self.failure_modes['phenotype_confused']
        if not base_failures:
            return self._random_case()
        
        base = random.choice(base_failures)
        
        with self.driver.session() as session:
            original = session.run("""
                MATCH (c:Case {case_id: $case_id}) RETURN c
            """, case_id=base['case_id']).single()
            
            if not original:
                return self._random_case()
            
            case_data = dict(original['c'])
            
            # Find gene with highest phenotype overlap that isn't the true gene
            confusing_gene = session.run("""
                MATCH (g:Gene)-[:EXPRESSED_IN_DISEASE]->(d:Disease)
                      -[:HAS_PHENOTYPE]->(h:HPO)
                WHERE h.hpo_id IN $patient_hpos
                  AND g.symbol <> $true_gene
                WITH g, count(h) as overlap
                ORDER BY overlap DESC
                LIMIT 1
                RETURN g.symbol, overlap
            """,
            patient_hpos=case_data.get('patient_hpo_ids', []),
            true_gene=base['causal_gene']).single()
            
            if not confusing_gene:
                return self._random_case()
            
            new_case = copy.deepcopy(case_data)
            new_case['case_id'] = f"adv_{uuid.uuid4().hex[:8]}"
            new_case['decoy_gene'] = confusing_gene['g.symbol']
            new_case['source'] = 'adversary_phenotype_amplified'
            new_case['difficulty_target'] = 'phenotype_confused'
            new_case['guaranteed_solvable'] = True
            
            self._persist_generated_case(new_case)
            return new_case
    
    def _persist_generated_case(self, case: dict):
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Case {case_id: $case_id})
                SET c += $props
                SET c.adversary_generated = true
                SET c.created_at = datetime()
            """, case_id=case['case_id'], props=case)
    
    def _random_case(self) -> dict:
        """Fallback: random case from the base dataset."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Case) WHERE c.adversary_generated IS NULL
                RETURN c ORDER BY rand() LIMIT 1
            """).single()
            return dict(result['c']) if result else {}
```

---

## PHASE 5: Multi-LLM Benchmark Runner (~30 min setup)

### Goal
Run your environment against 5 models on Groq, collect scores per tier,
output a comparison table. Store all results in Neo4j. This becomes your
README leaderboard.

### Task 5.1 — Write `benchmarks/multi_llm_runner.py`

```python
import os
import json
import time
from groq import Groq
from db.connection import get_driver
from rich.table import Table
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TEST = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
]

SYSTEM_PROMPT = """You are a clinical genomics AI agent navigating a variant knowledge graph.

You will receive:
- phenotypes_present: HPO terms the patient HAS
- phenotypes_absent: HPO terms strongly associated with candidates but patient does NOT have
- candidate_variants: variants at your current position with CADD scores and phenotype associations
- current_gene: your current position in the graph

Your task: identify which gene/variant is causal. Output ONLY valid JSON matching:
{
  "hop_to": "<gene_symbol>",
  "reasoning": "<cite specific HPO terms, explain why you include or exclude candidates>",
  "differential": [
    {"gene": "<symbol>", "confidence": <float>, "evidence": "<brief>"},
    ...  // up to 5 entries, confidences sum to 1.0
  ]
}

Resisting high-CADD distractors that lack phenotype support is critical.
Use phenotypes_absent to actively exclude candidates."""

def run_model_on_case(client: Groq, model: str, observation: dict, 
                       max_steps: int = 8) -> dict:
    """Run one model on one case, return episode result."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    steps = []
    total_reward = 0.0
    causal_found = False
    
    for step in range(max_steps):
        messages.append({
            "role": "user", 
            "content": f"Step {step+1}. Observation:\n{json.dumps(observation, indent=2)}"
        })
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            
            action = json.loads(raw)
            messages.append({"role": "assistant", "content": raw})
            
            # Score this step (simplified — integrate with your real reward.py)
            step_reward = score_action(action, observation)
            total_reward += step_reward
            
            steps.append({
                "step": step,
                "hop_to": action.get("hop_to"),
                "top_hypothesis": action.get("differential", [{}])[0].get("gene"),
                "reward": step_reward,
                "used_absent_phenotype": any(
                    hpo in action.get("reasoning", "")
                    for hpo in observation.get("phenotypes_absent", [])
                ),
            })
            
            # Check if causal gene found
            if action.get("hop_to") == observation.get("causal_gene"):
                causal_found = True
                break
            
            # Update observation for next step (simplified)
            observation = get_next_observation(action["hop_to"], observation)
            time.sleep(0.5)  # rate limiting
            
        except (json.JSONDecodeError, KeyError) as e:
            steps.append({"step": step, "error": str(e), "reward": -0.1})
            break
    
    return {
        "model": model,
        "causal_found": causal_found,
        "total_reward": round(total_reward, 3),
        "steps_taken": len(steps),
        "used_exclusion_reasoning": any(s.get("used_absent_phenotype") for s in steps),
        "steps": steps,
    }

def run_benchmark(cases_per_tier: int = 5):
    """Run all models on all tiers, store results in Neo4j, print table."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    driver = get_driver()
    results = []
    
    # Load cases by tier from Neo4j
    tiers = {
        "tier1_basic": load_cases("tier1", cases_per_tier, driver),
        "tier2_intermediate": load_cases("tier2", cases_per_tier, driver),
        "tier3_mismatch": load_cases("tier3_phenotype_mismatch", cases_per_tier, driver),
    }
    
    for model in MODELS_TO_TEST:
        print(f"\nRunning {model}...")
        model_results = {"model": model}
        
        for tier_name, cases in tiers.items():
            tier_rewards = []
            tier_solved = 0
            
            for case in cases:
                obs = build_observation_from_case(case)
                ep_result = run_model_on_case(client, model, obs)
                tier_rewards.append(ep_result["total_reward"])
                if ep_result["causal_found"]:
                    tier_solved += 1
                
                # Persist to Neo4j
                persist_benchmark_result(driver, model, tier_name, case, ep_result)
            
            model_results[f"{tier_name}_avg_reward"] = round(
                sum(tier_rewards)/len(tier_rewards), 3
            )
            model_results[f"{tier_name}_solve_rate"] = f"{tier_solved}/{len(cases)}"
        
        results.append(model_results)
        print(f"  Done. Tier3 mismatch avg: {model_results['tier3_mismatch_avg_reward']}")
    
    print_results_table(results)
    return results

def load_cases(tier: str, n: int, driver) -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case {tier: $tier})
            RETURN c ORDER BY rand() LIMIT $n
        """, tier=tier, n=n)
        return [dict(r['c']) for r in result]

def persist_benchmark_result(driver, model: str, tier: str, case: dict, result: dict):
    with driver.session() as session:
        session.run("""
            CREATE (b:BenchmarkRun {
                model: $model,
                tier: $tier,
                case_id: $case_id,
                causal_found: $causal_found,
                total_reward: $total_reward,
                steps_taken: $steps_taken,
                used_exclusion: $used_exclusion,
                timestamp: datetime()
            })
        """,
        model=model,
        tier=tier,
        case_id=case.get('case_id'),
        causal_found=result['causal_found'],
        total_reward=result['total_reward'],
        steps_taken=result['steps_taken'],
        used_exclusion=result['used_exclusion_reasoning'])

def print_results_table(results: list):
    console = Console()
    table = Table(title="Narada-Env Multi-LLM Benchmark", show_header=True)
    
    table.add_column("Model", style="bold cyan")
    table.add_column("Tier 1\nBasic", justify="center")
    table.add_column("Tier 2\nIntermediate", justify="center")
    table.add_column("Tier 3\nMismatch ⚠️", justify="center", style="bold yellow")
    table.add_column("Tier 1\nSolve%", justify="center")
    table.add_column("Tier 2\nSolve%", justify="center")
    table.add_column("Tier 3\nSolve%", justify="center", style="bold red")
    
    for r in sorted(results, key=lambda x: x.get('tier3_mismatch_avg_reward', 0), reverse=True):
        table.add_row(
            r['model'].split('-')[0] + '...',  # truncate for display
            str(r.get('tier1_basic_avg_reward', 'N/A')),
            str(r.get('tier2_intermediate_avg_reward', 'N/A')),
            str(r.get('tier3_mismatch_avg_reward', 'N/A')),
            r.get('tier1_basic_solve_rate', 'N/A'),
            r.get('tier2_intermediate_solve_rate', 'N/A'),
            r.get('tier3_mismatch_solve_rate', 'N/A'),
        )
    
    console.print(table)
    
    # Save as markdown for README
    with open("benchmark_results.md", "w") as f:
        f.write("## Multi-LLM Benchmark Results\n\n")
        f.write("| Model | Tier 1 | Tier 2 | Tier 3 (Mismatch) |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['model']} | "
                    f"{r.get('tier1_basic_avg_reward', 'N/A')} | "
                    f"{r.get('tier2_intermediate_avg_reward', 'N/A')} | "
                    f"{r.get('tier3_mismatch_avg_reward', 'N/A')} |\n")
    
    print("\nSaved to benchmark_results.md")

if __name__ == "__main__":
    run_benchmark(cases_per_tier=5)
```

---

## PHASE 6: Cognitive Load Analysis — Generates the Q2 Stat (~20 min)

### Goal
Produce the concrete number: "In a typical exome case, a clinician must manually
review N variants. Narada solves this in M hops." Run this on your Neo4j data.

### Task 6.1 — Write `analysis/cognitive_load.py`

```python
from db.connection import get_driver
import statistics

def compute_cognitive_load_stats():
    """
    For each case in the environment:
    1. Count variants with CADD > 20 (clinically reportable threshold)
    2. Of those, count how many match ≥1 patient phenotype
    3. That's the realistic manual review burden
    """
    driver = get_driver()
    
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            WITH c LIMIT 100
            MATCH (v:Variant)-[:ASSOCIATED_WITH_DISEASE]->(d:Disease)
                  -[:HAS_PHENOTYPE]->(h:HPO)
            WHERE v.cadd_score > 20
              AND h.hpo_id IN c.patient_hpo_ids
            WITH c.case_id as case_id, count(DISTINCT v) as phenotype_matching_reportable
            RETURN 
                avg(phenotype_matching_reportable) as avg_review_burden,
                min(phenotype_matching_reportable) as min_burden,
                max(phenotype_matching_reportable) as max_burden,
                percentileCont(phenotype_matching_reportable, 0.5) as median_burden
        """)
        
        stats = result.single()
        
        print(f"""
=== COGNITIVE LOAD ANALYSIS ===
Clinically reportable variants (CADD > 20) matching ≥1 patient phenotype:

  Average review burden: {stats['avg_review_burden']:.1f} variants per case
  Median:               {stats['median_burden']:.1f}
  Range:                {stats['min_burden']} – {stats['max_burden']}

This is the manual search space a clinical genomicist faces.
Narada agents navigate this in [CHECK YOUR EPISODE LOGS FOR AVG HOPS] steps.

Add this to your README problem statement.
        """)
        
        return dict(stats)

if __name__ == "__main__":
    compute_cognitive_load_stats()
```

---

## EXECUTION ORDER FOR CLAUDE CODE

Run phases in this exact order. Do not skip.

```
1.  python db/connection.py              # verify Neo4j connects
2.  python db/schema.py                  # create constraints & indexes
3.  python db/ingest.py                  # migrate graph data to cloud
4.  python db/verify.py                  # confirm data is queryable

5.  # Integrate observation_builder.py into your environment's step() function
6.  # Integrate reward.py score_exclusion_reasoning() into reward calculation
7.  # Update action schema in actions.py to require differential field
8.  # Integrate score_differential() into Overseer scoring

9.  python adversary/failure_analyzer.py # test failure mode loading
    # (will return empty on first run — that's expected)
    
10. python analysis/cognitive_load.py    # generate the Q2 stat

11. # Start a training run to generate failure logs in Neo4j
    # After 50+ episodes, re-run failure_analyzer to populate failure_modes

12. python benchmarks/multi_llm_runner.py  # run while training continues
    # This runs in ~30 min and generates benchmark_results.md
```

---

## WHAT TO TELL YOUR JUDGES

Once all phases are complete, your honest pitch is:

> **Architecture:** Two-agent system (Detective + Overseer) with a failure-mode
> adversary that generates solvable but challenging cases by amplifying documented
> failure patterns. Adversary component is grounded in PAIRED-style curriculum
> generation — challenging but solvable is a hard guarantee because the true causal
> path is never modified, only the distractor strength is increased.

> **What makes it different from graph search:** Three things graph search cannot do:
> (1) reason from absent phenotypes to exclude candidates,
> (2) maintain and update a ranked differential hypothesis across steps, and
> (3) be specifically penalised for anchoring on CADD score over phenotype fit —
> a known cognitive bias in clinical interpretation.

> **Compute constraint:** Qwen 1.7B on T4 is proof-of-concept for the training
> pipeline. GRPO on sub-2B models is documented to require dense reward signals
> at scale (consistent with G2RPO-A findings). The environment and curriculum
> design are the contribution — not the specific model checkpoint.

> **Scale:** 92,000 pathogenic variants, 28,000 disease nodes, all queryable
> from Neo4j AuraDB. The benchmark table shows differential performance across
> 5 LLMs on all three curriculum tiers — Tier 3 (phenotype_mismatch) is the
> discriminating column.

---

## FILES TO CREATE (summary)

```
db/
  connection.py
  schema.py
  ingest.py
  verify.py

environment/
  observation_builder.py   ← adds phenotypes_absent
  reward.py                ← adds score_exclusion_reasoning()
  actions.py               ← adds differential field to NaradaAction

adversary/
  failure_analyzer.py
  case_generator.py        ← FailureModeAdversary

benchmarks/
  multi_llm_runner.py

analysis/
  cognitive_load.py

.env                       ← never commit this
benchmark_results.md       ← auto-generated, goes in README
```

---

*End of prompt. Paste into Claude Code and begin with Phase 1.*
