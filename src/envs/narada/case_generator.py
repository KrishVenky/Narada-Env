"""
Narada: Patient case generator.

Builds episode cases from the knowledge graph + disease catalog.
Each case is a dict matching PatientCase structure.

Task types:
  monogenic          — single causal gene, 3-4 phenotypes, 5-8 candidates
  oligogenic         — 2-3 causal genes, 5-7 phenotypes, 10-15 candidates
  phenotype_mismatch — cardiac patient + high-pathogenicity cancer decoy
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import (
    DISEASE_CATALOG,
    GENE_TO_DISEASES,
    PATHWAY_MAP,
    NaradaGraph,
    _clinsig_to_score,
    _slugify,
)
from .models import GraphNode, Variant

# ── BRCA1/BRCA2 decoy pool ─────────────────────────────────────────────────────
# Frameshift/nonsense variants are maximally salient for LLMs — best decoys.
_DECOY_GENES = ["BRCA1", "BRCA2", "TP53", "MLH1", "MSH2"]
_DECOY_TYPES = {"frameshift", "deletion", "nonsense", "stop_gained", "indel"}


def _is_high_impact(v: Dict[str, Any]) -> bool:
    vtype = v.get("variant_type", "").lower()
    name = v.get("name", "").lower()
    return (
        any(t in vtype for t in _DECOY_TYPES)
        or "frameshift" in name
        or "stop" in name
        or "del" in vtype
    )


def _pick_variants(
    graph: NaradaGraph,
    genes: List[str],
    n: int,
    prefer_high_impact: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Pick up to n variants from the given gene list."""
    if rng is None:
        rng = random.Random()
    pool = graph.get_variants_for_genes(genes)
    if prefer_high_impact:
        high = [v for v in pool if _is_high_impact(v)]
        pool = high if high else pool
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def _variant_to_model(v: Dict[str, Any], graph: NaradaGraph) -> Variant:
    var_id = graph.variant_node_id(v["allele_id"])
    return Variant(
        id=var_id,
        allele_id=v["allele_id"],
        gene=v["gene"],
        name=v["name"][:150] if v["name"] else f"{v['gene']} variant",
        variant_type=v["variant_type"],
        clinical_significance=v["clnsig"],
        pathogenicity_score=_clinsig_to_score(v["clnsig"]),
        disease_associations=v["diseases"][:3],
    )


def _dict_to_graph_node(graph: NaradaGraph, node_id: str) -> GraphNode:
    nd = graph.get_node(node_id)
    if nd is None:
        return GraphNode(
            id=node_id, type="unknown", name=node_id,
            description="", connected_node_ids=[],
        )
    neighbors = graph.get_neighbors(node_id)
    return GraphNode(
        id=nd["id"],
        type=nd["type"],
        name=nd["name"],
        description=nd["description"],
        connected_node_ids=neighbors[:30],  # cap for observation size
        metadata=nd["metadata"],
    )


# ── Case structure ─────────────────────────────────────────────────────────────

class PatientCase:
    """
    A single patient episode definition.
    Immutable after construction — shared state lives in the environment.
    """

    def __init__(
        self,
        case_id: str,
        task_type: str,
        disease_name: str,
        causal_genes: List[str],
        causal_allele_ids: List[str],          # ground truth
        patient_hpo_ids: List[str],
        patient_phenotype_names: List[str],
        candidate_variants: List[Variant],
        starting_node_id: str,
        relevant_node_ids: Set[str],
        decoy_gene: Optional[str] = None,
        absent_hpo_ids: Optional[List[str]] = None,
        absent_phenotype_names: Optional[List[str]] = None,
    ) -> None:
        self.case_id = case_id
        self.task_type = task_type
        self.disease_name = disease_name
        self.causal_genes = causal_genes
        self.causal_allele_ids = causal_allele_ids
        self.patient_hpo_ids = patient_hpo_ids
        self.patient_phenotype_names = patient_phenotype_names
        self.absent_hpo_ids = absent_hpo_ids or []
        self.absent_phenotype_names = absent_phenotype_names or []
        self.candidate_variants = candidate_variants
        self.starting_node_id = starting_node_id
        self.relevant_node_ids = relevant_node_ids
        self.decoy_gene = decoy_gene

    @property
    def ground_truth_variant_ids(self) -> List[str]:
        return [f"VAR:{aid}" for aid in self.causal_allele_ids]


# ── Generators ─────────────────────────────────────────────────────────────────

def _pick_hpo_subset(
    hpo_ids: List[str],
    graph: NaradaGraph,
    n: int,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """Return (hpo_ids, names) for n terms, using only ones present in graph."""
    present = [h for h in hpo_ids if h in graph.nodes]
    if not present:
        present = hpo_ids[:n]
    chosen = rng.sample(present, min(n, len(present)))
    names = [graph.get_hpo_name(h) for h in chosen]
    return chosen, names


def _find_starting_node(
    graph: NaradaGraph,
    hpo_ids: List[str],
    rng: random.Random,
) -> str:
    """Find a good starting phenotype node in the graph."""
    for h in rng.sample(hpo_ids, len(hpo_ids)):
        if h in graph.nodes:
            return h
    # Fallback: any phenotype node
    pheno_nodes = [nid for nid, nd in graph.nodes.items() if nd["type"] == "phenotype"]
    return rng.choice(pheno_nodes) if pheno_nodes else list(graph.nodes.keys())[0]


def generate_monogenic_case(
    graph: NaradaGraph,
    rng: Optional[random.Random] = None,
) -> PatientCase:
    """Single causal gene, 3-4 phenotypes, 5-8 candidate variants."""
    if rng is None:
        rng = random.Random()

    eligible = [d for d in DISEASE_CATALOG if "monogenic" in d["task_types"] and d["genes"]]
    disease = rng.choice(eligible)

    # Pick one primary gene that has variants
    for gene in rng.sample(disease["genes"], len(disease["genes"])):
        if graph.get_variants_for_gene(gene):
            causal_gene = gene
            break
    else:
        raise RuntimeError(f"No variants found for any gene in {disease['disease']}")

    # Ground truth: pick 1-2 causal variants
    causal_raw = _pick_variants(graph, [causal_gene], n=2, prefer_high_impact=True, rng=rng)
    if not causal_raw:
        raise RuntimeError(f"No variants for {causal_gene}")
    causal_allele_ids = [v["allele_id"] for v in causal_raw]

    # Patient phenotypes: 3-4 terms
    n_pheno = rng.randint(3, 4)
    hpo_ids, hpo_names = _pick_hpo_subset(disease["hpo_ids"], graph, n_pheno, rng)

    # Absent phenotypes: disease HPO terms the patient does NOT have (diagnostic exclusions)
    chosen_set = set(hpo_ids)
    absent_candidates = [h for h in disease["hpo_ids"] if h not in chosen_set and h in graph.nodes]
    rng.shuffle(absent_candidates)
    absent_hpo_ids = absent_candidates[:3]
    absent_names = [graph.get_hpo_name(h) for h in absent_hpo_ids]

    # Candidate variants: causal + 3-6 distractors from same-pathway genes
    target_pathway = disease["pathway"]
    distractor_genes = [
        g for g in graph.gene_variants.keys()
        if g != causal_gene
        and (
            PATHWAY_MAP.get(g) == target_pathway
            or any(target_pathway == d["pathway"] for d in GENE_TO_DISEASES.get(g, []))
        )
    ]
    if len(distractor_genes) < 3:
        distractor_genes = [g for g in graph.gene_variants.keys() if g != causal_gene]
    n_distractors = rng.randint(3, 6)
    distractor_raw = _pick_variants(
        graph,
        rng.sample(distractor_genes, min(6, len(distractor_genes))),
        n=n_distractors,
        rng=rng,
    )
    all_raw = causal_raw + distractor_raw
    rng.shuffle(all_raw)
    candidates = [_variant_to_model(v, graph) for v in all_raw]

    starting_node = _find_starting_node(graph, hpo_ids, rng)
    relevant = graph.relevant_nodes_for_case(causal_genes=[causal_gene], patient_hpo_ids=hpo_ids)

    return PatientCase(
        case_id=str(uuid.uuid4())[:8],
        task_type="monogenic",
        disease_name=disease["disease"],
        causal_genes=[causal_gene],
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=hpo_ids,
        patient_phenotype_names=hpo_names,
        candidate_variants=candidates,
        starting_node_id=starting_node,
        relevant_node_ids=relevant,
        absent_hpo_ids=absent_hpo_ids,
        absent_phenotype_names=absent_names,
    )


def generate_oligogenic_case(
    graph: NaradaGraph,
    rng: Optional[random.Random] = None,
) -> PatientCase:
    """2-3 causal genes, 5-7 phenotypes, 10-15 candidates."""
    if rng is None:
        rng = random.Random()

    eligible = [d for d in DISEASE_CATALOG if "oligogenic" in d["task_types"] and len(d["genes"]) >= 2]
    disease = rng.choice(eligible)

    # Pick 2 causal genes that have variants
    causal_genes = []
    for gene in rng.sample(disease["genes"], len(disease["genes"])):
        if graph.get_variants_for_gene(gene):
            causal_genes.append(gene)
        if len(causal_genes) == 2:
            break

    if len(causal_genes) < 2:
        causal_genes = [g for g in disease["genes"] if graph.get_variants_for_gene(g)][:2]
    if not causal_genes:
        raise RuntimeError(f"No variants for genes in {disease['disease']}")

    # 1-2 causal variants per gene
    causal_raw = []
    causal_allele_ids = []
    for gene in causal_genes:
        vs = _pick_variants(graph, [gene], n=2, prefer_high_impact=True, rng=rng)
        causal_raw.extend(vs)
        causal_allele_ids.extend(v["allele_id"] for v in vs)

    # Patient phenotypes: 5-7 terms
    n_pheno = rng.randint(5, min(7, len(disease["hpo_ids"])))
    hpo_ids, hpo_names = _pick_hpo_subset(disease["hpo_ids"], graph, n_pheno, rng)

    # Absent phenotypes
    chosen_set = set(hpo_ids)
    absent_candidates = [h for h in disease["hpo_ids"] if h not in chosen_set and h in graph.nodes]
    rng.shuffle(absent_candidates)
    absent_hpo_ids = absent_candidates[:3]
    absent_names = [graph.get_hpo_name(h) for h in absent_hpo_ids]

    # Distractors: from same-pathway genes
    target_pathway = disease["pathway"]
    distractor_genes = [
        g for g in graph.gene_variants.keys()
        if g not in causal_genes
        and (
            PATHWAY_MAP.get(g) == target_pathway
            or any(target_pathway == d["pathway"] for d in GENE_TO_DISEASES.get(g, []))
        )
    ]
    if len(distractor_genes) < 4:
        distractor_genes = [g for g in graph.gene_variants.keys() if g not in causal_genes]
    n_distractors = rng.randint(6, 9)
    distractor_raw = _pick_variants(
        graph,
        rng.sample(distractor_genes, min(6, len(distractor_genes))),
        n=n_distractors,
        rng=rng,
    )

    all_raw = causal_raw + distractor_raw
    rng.shuffle(all_raw)
    candidates = [_variant_to_model(v, graph) for v in all_raw[:15]]

    starting_node = _find_starting_node(graph, hpo_ids, rng)
    relevant = graph.relevant_nodes_for_case(causal_genes=causal_genes, patient_hpo_ids=hpo_ids)

    return PatientCase(
        case_id=str(uuid.uuid4())[:8],
        task_type="oligogenic",
        disease_name=disease["disease"],
        causal_genes=causal_genes,
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=hpo_ids,
        patient_phenotype_names=hpo_names,
        candidate_variants=candidates,
        starting_node_id=starting_node,
        relevant_node_ids=relevant,
        absent_hpo_ids=absent_hpo_ids,
        absent_phenotype_names=absent_names,
    )


def generate_mismatch_case(
    graph: NaradaGraph,
    rng: Optional[random.Random] = None,
) -> PatientCase:
    """
    Phenotype mismatch: cardiac/neurological patient with high-pathogenicity
    cancer decoy in the candidate pool. Tests causal discipline.
    """
    if rng is None:
        rng = random.Random()

    eligible = [d for d in DISEASE_CATALOG if "phenotype_mismatch" in d["task_types"]]
    disease = rng.choice(eligible)

    # Causal gene from actual disease
    for gene in rng.sample(disease["genes"], len(disease["genes"])):
        if graph.get_variants_for_gene(gene):
            causal_gene = gene
            break
    else:
        raise RuntimeError(f"No variants for {disease['disease']}")

    # Causal variants
    causal_raw = _pick_variants(graph, [causal_gene], n=2, prefer_high_impact=True, rng=rng)
    causal_allele_ids = [v["allele_id"] for v in causal_raw]

    # Patient phenotypes: 4-6 terms
    n_pheno = rng.randint(4, min(6, len(disease["hpo_ids"])))
    hpo_ids, hpo_names = _pick_hpo_subset(disease["hpo_ids"], graph, n_pheno, rng)

    # Absent phenotypes
    chosen_set = set(hpo_ids)
    absent_candidates = [h for h in disease["hpo_ids"] if h not in chosen_set and h in graph.nodes]
    rng.shuffle(absent_candidates)
    absent_hpo_ids = absent_candidates[:3]
    absent_names = [graph.get_hpo_name(h) for h in absent_hpo_ids]

    # DECOY: pick a high-pathogenicity BRCA1/BRCA2 frameshift
    decoy_gene = rng.choice([g for g in _DECOY_GENES if graph.get_variants_for_gene(g)])
    decoy_raw = _pick_variants(graph, [decoy_gene], n=2, prefer_high_impact=True, rng=rng)
    # Boost decoy pathogenicity score for maximum LLM temptation
    for v in decoy_raw:
        v = dict(v)  # copy to avoid mutating global cache

    # Same-pathway distractors
    target_pathway = disease["pathway"]
    distractor_genes = [
        g for g in graph.gene_variants.keys()
        if g != causal_gene and g not in _DECOY_GENES
        and (
            PATHWAY_MAP.get(g) == target_pathway
            or any(target_pathway == d["pathway"] for d in GENE_TO_DISEASES.get(g, []))
        )
    ]
    if len(distractor_genes) < 2:
        distractor_genes = [
            g for g in graph.gene_variants.keys()
            if g != causal_gene and g not in _DECOY_GENES
        ]
    n_distractors = rng.randint(3, 5)
    distractor_raw = _pick_variants(
        graph,
        rng.sample(distractor_genes, min(4, len(distractor_genes))),
        n=n_distractors,
        rng=rng,
    )

    all_raw = causal_raw + decoy_raw + distractor_raw
    rng.shuffle(all_raw)
    candidates = [_variant_to_model(v, graph) for v in all_raw[:15]]

    starting_node = _find_starting_node(graph, hpo_ids, rng)
    relevant = graph.relevant_nodes_for_case(causal_genes=[causal_gene], patient_hpo_ids=hpo_ids)

    return PatientCase(
        case_id=str(uuid.uuid4())[:8],
        task_type="phenotype_mismatch",
        disease_name=disease["disease"],
        causal_genes=[causal_gene],
        causal_allele_ids=causal_allele_ids,
        patient_hpo_ids=hpo_ids,
        patient_phenotype_names=hpo_names,
        candidate_variants=candidates,
        starting_node_id=starting_node,
        relevant_node_ids=relevant,
        decoy_gene=decoy_gene,
        absent_hpo_ids=absent_hpo_ids,
        absent_phenotype_names=absent_names,
    )


_GENERATORS = {
    "monogenic": generate_monogenic_case,
    "oligogenic": generate_oligogenic_case,
    "phenotype_mismatch": generate_mismatch_case,
}

MAX_STEPS = {
    "monogenic": 15,
    "oligogenic": 25,
    "phenotype_mismatch": 20,
}


def generate_case(
    graph: NaradaGraph,
    task_type: str,
    seed: Optional[int] = None,
) -> PatientCase:
    if task_type not in _GENERATORS:
        raise ValueError(f"Unknown task_type: {task_type!r}. Choose from {list(_GENERATORS)}")
    rng = random.Random(seed)
    return _GENERATORS[task_type](graph, rng=rng)
