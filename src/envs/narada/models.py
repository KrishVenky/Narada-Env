"""
Narada: Pydantic v2 data models.

All observation/action/state models live here.
No imports from server/ — models are shared between client, server, and inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Graph primitives ──────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    type: str  # gene | variant | phenotype | disease | pathway
    name: str
    description: str
    connected_node_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Variant(BaseModel):
    id: str            # e.g. "VAR:15041"
    allele_id: str     # raw ClinVar AlleleID
    gene: str          # e.g. "BRCA1"
    name: str          # HGVS name
    variant_type: str  # Deletion | Insertion | SNV | Indel ...
    clinical_significance: str  # Pathogenic | Likely pathogenic | ...
    pathogenicity_score: float  # 0.0–1.0 derived from clnsig
    disease_associations: List[str]  # disease names from PhenotypeList


# ── Action ────────────────────────────────────────────────────────────────────

class NaradaAction(BaseModel):
    action_type: str  # hop | flag_causal | request_lab | backtrack | summarise_trail
    node_id: Optional[str] = None      # target for hop
    variant_id: Optional[str] = None   # target for flag_causal
    test_type: Optional[str] = None    # test label for request_lab
    reasoning: str = ""


# ── Observation ───────────────────────────────────────────────────────────────

class NaradaObservation(BaseModel):
    step: int
    max_steps: int
    task_type: str         # monogenic | oligogenic | phenotype_mismatch
    current_node: GraphNode
    trail: List[GraphNode] = Field(default_factory=list)
    patient_phenotypes: List[str]      # HPO term IDs e.g. ["HP:0001250"]
    phenotype_names: List[str]         # human-readable, parallel to patient_phenotypes
    candidate_variants: List[Variant]  # 5–20 variants to choose from
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ── Step result (server → client) ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: NaradaObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── State metadata ─────────────────────────────────────────────────────────────

class NaradaState(BaseModel):
    episode_id: str
    task_type: str
    case_id: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
    flagged_variants: List[str] = Field(default_factory=list)
    ground_truth_variants: List[str] = Field(default_factory=list)
