"""
ClinDetect: Knowledge graph builder.

Builds the navigation graph from:
  1. data/hp.obo            — HPO phenotype terms and hierarchy
  2. data/clinvar_pathogenic.tsv — filtered ClinVar variants

Graph schema
------------
Node types:
  phenotype  — HP:XXXXXXX terms
  disease    — disease name derived from ClinVar PhenotypeList
  gene       — gene symbol (MYH7, BRCA1, …)
  variant    — individual ClinVar variant
  pathway    — coarse pathway group (cardiac, neurological, …)

Edge direction is bidirectional (undirected for navigation).
Stored as: graph["edges"][node_id] = [list of connected node_ids]
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).parent.parent.parent.parent.parent / "data",  # /app/data in Docker
        Path(__file__).parent.parent.parent.parent / "data",
        Path(__file__).parent.parent.parent / "data",
        Path.cwd() / "data",
    ]
    for p in candidates:
        if p.is_dir() and (p / "hp.obo").exists():
            return p
    return candidates[0]


# ── Pathway classification ─────────────────────────────────────────────────────

PATHWAY_MAP: Dict[str, str] = {
    # Cardiac
    "MYH7": "cardiac", "MYBPC3": "cardiac", "MYH6": "cardiac",
    "TNNT2": "cardiac", "TNNI3": "cardiac", "TPM1": "cardiac",
    "TTN": "cardiac", "LMNA": "cardiac", "SCN5A": "cardiac",
    "KCNQ1": "cardiac", "KCNH2": "cardiac", "PLN": "cardiac",
    "RYR2": "cardiac", "DSP": "cardiac", "PKP2": "cardiac",
    "JUP": "cardiac", "DSG2": "cardiac", "DSC2": "cardiac",
    # Neurological
    "SCN1A": "neurological", "MECP2": "neurological", "PTEN": "neurological",
    "TSC1": "neurological", "TSC2": "neurological", "FMR1": "neurological",
    "DMPK": "neurological", "HTT": "neurological", "ATXN1": "neurological",
    "ATXN2": "neurological", "ATXN3": "neurological", "SNCA": "neurological",
    "LRRK2": "neurological", "PARK2": "neurological", "GBA1": "neurological",
    # Metabolic
    "PAH": "metabolic", "PCSK9": "metabolic", "LDLR": "metabolic",
    "APOB": "metabolic", "HMGCR": "metabolic", "GBA1": "metabolic",
    "HEXA": "metabolic", "HEXB": "metabolic", "GALC": "metabolic",
    "ARSA": "metabolic", "ATP7B": "metabolic", "SLC25A13": "metabolic",
    # Cancer (used as decoys in cardiac/neuro tasks)
    "BRCA1": "cancer", "BRCA2": "cancer", "TP53": "cancer",
    "MLH1": "cancer", "MSH2": "cancer", "MSH6": "cancer",
    "APC": "cancer", "RB1": "cancer", "VHL": "cancer",
    "PTEN": "cancer", "CDH1": "cancer", "STK11": "cancer",
    # Connective tissue
    "FBN1": "connective_tissue", "FBN2": "connective_tissue",
    "COL1A1": "connective_tissue", "COL1A2": "connective_tissue",
    "COL3A1": "connective_tissue", "ELN": "connective_tissue",
    # Pulmonary
    "CFTR": "pulmonary",
    # Renal
    "PKD1": "renal", "PKD2": "renal", "PKHD1": "renal",
    # Musculoskeletal
    "DMD": "musculoskeletal", "DYSF": "musculoskeletal",
    "CAPN3": "musculoskeletal", "ANO5": "musculoskeletal",
    "PHEX": "musculoskeletal", "ALPL": "musculoskeletal",
    # Ophthalmology
    "ABCA4": "ophthalmology", "USH2A": "ophthalmology",
    "MYO7A": "ophthalmology", "CRB1": "ophthalmology",
    "RPE65": "ophthalmology", "RPGR": "ophthalmology",
    # Haematology
    "HBB": "haematology", "HBA1": "haematology", "HBA2": "haematology",
    "F8": "haematology", "F9": "haematology", "VWF": "haematology",
    "G6PD": "haematology",
    # Immunology
    "RAG1": "immunology", "RAG2": "immunology", "ADA": "immunology",
    "BTK": "immunology", "CYBB": "immunology",
    # Endocrine
    "KCNJ11": "endocrine", "ABCC8": "endocrine", "GCK": "endocrine",
    "HNF1A": "endocrine", "INS": "endocrine",
}

# Pathogenicity score by significance string
_PATHOGENICITY_SCORES: Dict[str, float] = {
    "pathogenic": 0.95,
    "likely pathogenic": 0.75,
    "pathogenic/likely pathogenic": 0.85,
}


def _clinsig_to_score(clnsig: str) -> float:
    low = clnsig.lower()
    for key, score in sorted(_PATHOGENICITY_SCORES.items(), key=lambda x: -x[1]):
        if key in low:
            return score
    return 0.7  # fallback


# ── HPO parser ────────────────────────────────────────────────────────────────

def parse_hpo_obo(obo_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse hp.obo into a dict: HP:XXXXXXX → {name, parents, synonyms, def}.
    Only parses [Term] stanzas, skips obsoletes.
    """
    terms: Dict[str, Dict[str, Any]] = {}
    current: Optional[Dict[str, Any]] = None

    with open(obo_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if current and not current.get("is_obsolete"):
                    terms[current["id"]] = current
                current = {"id": "", "name": "", "parents": [], "synonyms": [], "def": ""}
                continue
            if line.startswith("[") and current:
                if not current.get("is_obsolete"):
                    terms[current["id"]] = current
                current = None
                continue
            if current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = line[6:].strip()
            elif line.startswith("def: "):
                # Strip quotes and references
                m = re.match(r'def: "([^"]*)"', line)
                current["def"] = m.group(1) if m else ""
            elif line.startswith("is_a: "):
                parent_id = line[6:].split("!")[0].strip()
                current["parents"].append(parent_id)
            elif line.startswith("synonym: "):
                m = re.match(r'synonym: "([^"]*)"', line)
                if m:
                    current["synonyms"].append(m.group(1))
            elif line.startswith("is_obsolete: true"):
                current["is_obsolete"] = True

    if current and not current.get("is_obsolete"):
        terms[current["id"]] = current

    logger.info("Parsed %d HPO terms from %s", len(terms), obo_path)
    return terms


# ── ClinVar loader ─────────────────────────────────────────────────────────────

def load_clinvar_variants(
    tsv_path: Path,
    max_per_gene: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load clinvar_pathogenic.tsv.
    Returns dict: gene_symbol → [list of variant dicts], capped at max_per_gene.
    """
    gene_variants: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Set[str] = set()

    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("GeneSymbol", "").strip()
            allele_id = row.get("#AlleleID", "").strip()
            if not gene or allele_id in seen:
                continue
            seen.add(allele_id)

            diseases_raw = row.get("PhenotypeList", "")
            diseases = [
                d.strip()
                for d in re.split(r"[|;]", diseases_raw)
                if d.strip() and d.strip().lower() not in ("not provided", "-", "")
            ]
            if not diseases:
                continue

            gene_variants[gene].append({
                "allele_id": allele_id,
                "gene": gene,
                "name": row.get("Name", "").strip(),
                "variant_type": row.get("Type", "").strip(),
                "clnsig": row.get("ClinicalSignificance", "").strip(),
                "diseases": diseases,
                "chromosome": row.get("Chromosome", "").strip(),
                "start": row.get("Start", "").strip(),
            })

    # Cap per gene and keep first (already deduped, sorted by file order)
    result = {
        gene: variants[:max_per_gene]
        for gene, variants in gene_variants.items()
    }
    logger.info(
        "Loaded variants for %d genes (%d total)",
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


# ── Graph builder ─────────────────────────────────────────────────────────────

class ClinDetectGraph:
    """
    In-memory knowledge graph for ClinDetect episodes.

    nodes: dict[node_id → {id, type, name, description, metadata}]
    edges: dict[node_id → set(connected_node_ids)]
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.hpo_terms: Dict[str, Dict[str, Any]] = {}
        self.gene_variants: Dict[str, List[Dict[str, Any]]] = {}
        self._pathway_nodes: Dict[str, str] = {}  # pathway_name → node_id
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, data_dir: Optional[Path] = None) -> None:
        if self._loaded:
            return
        if data_dir is None:
            data_dir = _find_data_dir()

        obo_path = data_dir / "hp.obo"
        tsv_path = data_dir / "clinvar_pathogenic.tsv"

        if not obo_path.exists():
            raise FileNotFoundError(f"hp.obo not found at {obo_path}")
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"clinvar_pathogenic.tsv not found at {tsv_path}. "
                "Run scripts/filter_clinvar.py first."
            )

        self.hpo_terms = parse_hpo_obo(obo_path)
        self.gene_variants = load_clinvar_variants(tsv_path)

        self._build_graph()
        self._loaded = True
        logger.info(
            "Graph loaded: %d nodes, %d edge-pairs",
            len(self.nodes),
            sum(len(v) for v in self.edges.values()) // 2,
        )

    def _build_graph(self) -> None:
        # 1. Pathway nodes
        for pathway in set(PATHWAY_MAP.values()):
            pid = f"PATH:{pathway}"
            self._add_node(pid, "pathway", pathway.replace("_", " ").title(), f"{pathway} pathway")
            self._pathway_nodes[pathway] = pid

        # 2. Gene nodes + variant nodes
        for gene, variants in self.gene_variants.items():
            gene_id = f"GENE:{gene}"
            pathway = PATHWAY_MAP.get(gene, "other")
            self._add_node(
                gene_id, "gene", gene,
                f"{gene} gene — {pathway} pathway",
                {"pathway": pathway, "variant_count": len(variants)},
            )
            # Gene ↔ pathway
            pathway_node = self._pathway_nodes.get(pathway)
            if pathway_node:
                self._add_edge(gene_id, pathway_node)

            # Variant nodes
            disease_set: Set[str] = set()
            for v in variants:
                var_id = f"VAR:{v['allele_id']}"
                score = _clinsig_to_score(v["clnsig"])
                self._add_node(
                    var_id, "variant",
                    f"{gene}:{v['variant_type']}",
                    v["name"][:120] if v["name"] else f"{gene} variant",
                    {
                        "gene": gene,
                        "allele_id": v["allele_id"],
                        "variant_type": v["variant_type"],
                        "clnsig": v["clnsig"],
                        "pathogenicity_score": score,
                        "diseases": v["diseases"][:3],
                    },
                )
                # Variant ↔ gene
                self._add_edge(var_id, gene_id)
                for d in v["diseases"][:3]:
                    disease_set.add(d)

            # Disease nodes (from variant disease associations)
            for disease_name in disease_set:
                dis_id = f"DIS:{_slugify(disease_name)}"
                self._add_node(
                    dis_id, "disease", disease_name,
                    f"Disease: {disease_name}",
                    {"gene": gene, "pathway": pathway},
                )
                self._add_edge(gene_id, dis_id)

        # 3. HPO phenotype nodes + edges to diseases
        # Only add HPO terms that appear in the disease catalog or have children/parents
        # that connect to disease nodes
        self._add_hpo_nodes()

    def _add_hpo_nodes(self) -> None:
        """
        Add HPO phenotype nodes and wire them into the graph.

        Strategy (fast, O(N+M) not O(N*M)):
        1. Add only HPO terms that appear in DISEASE_CATALOG + their ancestors (up to 5 levels)
        2. Wire phenotype → parent phenotype via HPO hierarchy
        3. Wire catalog HPO IDs directly to their disease nodes (explicit, no fuzzy matching)
        4. Build inverted word index for disease nodes and use it for lightweight matching
        """
        # Step 1: collect catalog HPO IDs + ancestors (to keep graph navigable)
        catalog_hpo_ids: Set[str] = set()
        for entry in DISEASE_CATALOG:
            for hpo_id in entry["hpo_ids"]:
                catalog_hpo_ids.add(hpo_id)
                # Walk ancestors up to 5 levels
                queue = list(self.hpo_terms.get(hpo_id, {}).get("parents", []))
                for _ in range(5):
                    next_q: List[str] = []
                    for pid in queue:
                        if pid.startswith("HP:"):
                            catalog_hpo_ids.add(pid)
                            next_q.extend(self.hpo_terms.get(pid, {}).get("parents", []))
                    queue = next_q
                    if not queue:
                        break

        # Step 2: add phenotype nodes (only catalog set + ancestors)
        for hpo_id in catalog_hpo_ids:
            term = self.hpo_terms.get(hpo_id)
            if not term:
                continue
            self._add_node(
                hpo_id, "phenotype", term["name"],
                term.get("def", term["name"]),
                {"parents": term.get("parents", [])},
            )

        # Step 3: wire phenotype hierarchy edges
        for hpo_id in catalog_hpo_ids:
            term = self.hpo_terms.get(hpo_id)
            if not term:
                continue
            for parent_id in term.get("parents", []):
                if parent_id in self.nodes:
                    self._add_edge(hpo_id, parent_id)

        # Step 4: explicit catalog wiring — phenotype → disease node
        # This is O(len(DISEASE_CATALOG) * avg_hpo_per_disease) ≈ O(100), not O(N*M)
        disease_name_index: Dict[str, str] = {}  # name_slug → node_id
        for nid, nd in self.nodes.items():
            if nd["type"] == "disease":
                disease_name_index[nd["name"].lower()] = nid

        for entry in DISEASE_CATALOG:
            # Find all disease nodes associated with this catalog entry's genes
            for gene in entry["genes"]:
                gene_id = f"GENE:{gene}"
                if gene_id not in self.nodes:
                    continue
                # All disease nodes linked to this gene
                gene_disease_nodes = [
                    nid for nid in self.edges.get(gene_id, set())
                    if self.nodes.get(nid, {}).get("type") == "disease"
                ]
                # Wire each catalog HPO term → all gene-associated disease nodes
                for hpo_id in entry["hpo_ids"]:
                    if hpo_id in self.nodes:
                        for dis_nid in gene_disease_nodes:
                            self._add_edge(hpo_id, dis_nid)

        # Step 5: lightweight inverted-index matching for broader coverage
        # Build word → disease_node_ids index (only once, O(M*W))
        word_to_diseases: Dict[str, List[str]] = defaultdict(list)
        for nid, nd in self.nodes.items():
            if nd["type"] != "disease":
                continue
            words = [w for w in nd["name"].lower().split() if len(w) > 4]
            for w in words:
                word_to_diseases[w].append(nid)

        # For each catalog HPO term, look up matching disease nodes via the index
        for hpo_id in catalog_hpo_ids:
            if hpo_id not in self.nodes:
                continue
            term = self.hpo_terms.get(hpo_id, {})
            hpo_words = [w for w in term.get("name", "").lower().split() if len(w) > 4]
            matched: Set[str] = set()
            for w in hpo_words:
                for dis_nid in word_to_diseases.get(w, []):
                    matched.add(dis_nid)
            for dis_nid in matched:
                self._add_edge(hpo_id, dis_nid)

    # ── Graph primitives ─────────────────────────────────────────────────────

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "name": name,
                "description": description,
                "metadata": metadata or {},
            }

    def _add_edge(self, a: str, b: str) -> None:
        if a != b and a in self.nodes and b in self.nodes:
            self.edges[a].add(b)
            self.edges[b].add(a)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        return sorted(self.edges.get(node_id, set()))

    def get_variants_for_gene(self, gene: str) -> List[Dict[str, Any]]:
        return self.gene_variants.get(gene, [])

    def get_variants_for_genes(self, genes: List[str]) -> List[Dict[str, Any]]:
        out = []
        for g in genes:
            out.extend(self.gene_variants.get(g, []))
        return out

    def get_gene_node_id(self, gene: str) -> Optional[str]:
        nid = f"GENE:{gene}"
        return nid if nid in self.nodes else None

    def get_hpo_name(self, hpo_id: str) -> str:
        term = self.hpo_terms.get(hpo_id)
        if term:
            return term["name"]
        node = self.nodes.get(hpo_id)
        return node["name"] if node else hpo_id

    def phenotype_node(self, hpo_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(hpo_id)

    def variant_node_id(self, allele_id: str) -> str:
        return f"VAR:{allele_id}"

    # ── Relevance scoring ─────────────────────────────────────────────────────

    def relevant_nodes_for_case(
        self,
        causal_genes: List[str],
        patient_hpo_ids: List[str],
    ) -> Set[str]:
        """
        Returns node IDs considered 'on-path' for a given case.
        Used by the environment to compute step-level rewards.
        """
        relevant: Set[str] = set()

        # Causal genes and their variants/diseases
        for gene in causal_genes:
            gene_id = f"GENE:{gene}"
            if gene_id in self.nodes:
                relevant.add(gene_id)
                # All variants of this gene
                for v in self.gene_variants.get(gene, []):
                    relevant.add(f"VAR:{v['allele_id']}")
                # All disease nodes linked to this gene
                for nid in self.edges.get(gene_id, set()):
                    if self.nodes.get(nid, {}).get("type") == "disease":
                        relevant.add(nid)
                # Pathway node
                pathway = self.nodes[gene_id]["metadata"].get("pathway")
                if pathway and pathway in self._pathway_nodes:
                    relevant.add(self._pathway_nodes[pathway])

        # Patient phenotype nodes and their ancestors (up to 3 levels)
        for hpo_id in patient_hpo_ids:
            relevant.add(hpo_id)
            # Walk up HPO hierarchy
            queue = list(self.hpo_terms.get(hpo_id, {}).get("parents", []))
            for _ in range(3):
                next_queue = []
                for pid in queue:
                    if pid in self.nodes:
                        relevant.add(pid)
                        next_queue.extend(self.hpo_terms.get(pid, {}).get("parents", []))
                queue = next_queue

        return relevant


# ── Disease → HPO term catalog (curated, embedded) ───────────────────────────
# Maps a canonical disease name to its known HPO phenotype IDs and gene(s).
# Used by case_generator.py to build patient cases.

DISEASE_CATALOG: List[Dict[str, Any]] = [
    # ── Cardiac ──────────────────────────────────────────────────────────────
    {
        "disease": "Hypertrophic cardiomyopathy",
        "genes": ["MYH7", "MYBPC3"],
        "hpo_ids": [
            "HP:0001639",  # Hypertrophic cardiomyopathy
            "HP:0001640",  # Cardiomegaly
            "HP:0004308",  # Ventricular arrhythmia
            "HP:0001685",  # Myocardial fibrosis
            "HP:0001644",  # Dilated cardiomyopathy (related)
            "HP:0004749",  # Atrial fibrillation
        ],
        "pathway": "cardiac",
        "task_types": ["monogenic", "oligogenic"],
    },
    {
        "disease": "Long QT syndrome",
        "genes": ["KCNQ1", "KCNH2", "SCN5A"],
        "hpo_ids": [
            "HP:0001657",  # Prolonged QT interval
            "HP:0004749",  # Atrial fibrillation
            "HP:0004308",  # Ventricular arrhythmia
            "HP:0001663",  # Ventricular fibrillation
            "HP:0001297",  # Stroke
        ],
        "pathway": "cardiac",
        "task_types": ["monogenic", "phenotype_mismatch"],
    },
    {
        "disease": "Dilated cardiomyopathy",
        "genes": ["TTN", "LMNA"],
        "hpo_ids": [
            "HP:0001644",  # Dilated cardiomyopathy
            "HP:0001640",  # Cardiomegaly
            "HP:0001638",  # Cardiomyopathy
            "HP:0004308",  # Ventricular arrhythmia
            "HP:0001671",  # Abnormal cardiac septum morphology
        ],
        "pathway": "cardiac",
        "task_types": ["monogenic", "oligogenic"],
    },
    # ── Neurological ─────────────────────────────────────────────────────────
    {
        "disease": "Dravet syndrome",
        "genes": ["SCN1A"],
        "hpo_ids": [
            "HP:0001250",  # Seizures
            "HP:0001263",  # Global developmental delay
            "HP:0000729",  # Autistic behavior
            "HP:0002194",  # Delayed gross motor development
            "HP:0001252",  # Hypotonia
        ],
        "pathway": "neurological",
        "task_types": ["monogenic", "phenotype_mismatch"],
    },
    {
        "disease": "Rett syndrome",
        "genes": ["MECP2"],
        "hpo_ids": [
            "HP:0001250",  # Seizures
            "HP:0002376",  # Developmental regression
            "HP:0001263",  # Global developmental delay
            "HP:0000729",  # Autistic behavior
            "HP:0002878",  # Respiratory failure
        ],
        "pathway": "neurological",
        "task_types": ["monogenic"],
    },
    {
        "disease": "Tuberous sclerosis complex",
        "genes": ["TSC1", "TSC2"],
        "hpo_ids": [
            "HP:0001250",  # Seizures
            "HP:0009716",  # Subependymal nodules
            "HP:0001263",  # Global developmental delay
            "HP:0010804",  # Tonic seizures
            "HP:0001646",  # Abnormal aortic morphology
        ],
        "pathway": "neurological",
        "task_types": ["oligogenic"],
    },
    # ── Metabolic ─────────────────────────────────────────────────────────────
    {
        "disease": "Phenylketonuria",
        "genes": ["PAH"],
        "hpo_ids": [
            "HP:0001249",  # Intellectual disability
            "HP:0001263",  # Global developmental delay
            "HP:0001250",  # Seizures
            "HP:0000729",  # Autistic behavior
            "HP:0001256",  # Intellectual disability, mild
        ],
        "pathway": "metabolic",
        "task_types": ["monogenic"],
    },
    {
        "disease": "Gaucher disease type 1",
        "genes": ["GBA1"],
        "hpo_ids": [
            "HP:0001744",  # Splenomegaly
            "HP:0001903",  # Anaemia
            "HP:0001873",  # Thrombocytopenia
            "HP:0002240",  # Hepatomegaly
            "HP:0010885",  # Avascular necrosis
        ],
        "pathway": "metabolic",
        "task_types": ["monogenic"],
    },
    {
        "disease": "Tay-Sachs disease",
        "genes": ["HEXA"],
        "hpo_ids": [
            "HP:0001250",  # Seizures
            "HP:0001249",  # Intellectual disability
            "HP:0001263",  # Global developmental delay
            "HP:0000365",  # Hearing loss
            "HP:0000486",  # Strabismus
        ],
        "pathway": "metabolic",
        "task_types": ["monogenic", "phenotype_mismatch"],
    },
    {
        "disease": "Wilson disease",
        "genes": ["ATP7B"],
        "hpo_ids": [
            "HP:0001638",  # Cardiomyopathy
            "HP:0002480",  # Hepatic encephalopathy
            "HP:0001410",  # Decreased liver function
            "HP:0001871",  # Abnormality of blood and blood-forming tissues
            "HP:0003128",  # Lactic acidosis
        ],
        "pathway": "metabolic",
        "task_types": ["monogenic"],
    },
    # ── Pulmonary ─────────────────────────────────────────────────────────────
    {
        "disease": "Cystic fibrosis",
        "genes": ["CFTR"],
        "hpo_ids": [
            "HP:0002099",  # Asthma
            "HP:0002110",  # Bronchiectasis
            "HP:0001738",  # Exocrine pancreatic insufficiency
            "HP:0003763",  # Meconium ileus
            "HP:0001891",  # Iron deficiency anaemia
        ],
        "pathway": "pulmonary",
        "task_types": ["monogenic"],
    },
    # ── Renal ─────────────────────────────────────────────────────────────────
    {
        "disease": "Autosomal dominant polycystic kidney disease",
        "genes": ["PKD1", "PKD2"],
        "hpo_ids": [
            "HP:0000113",  # Polycystic kidney dysplasia
            "HP:0001410",  # Decreased liver function
            "HP:0002240",  # Hepatomegaly
            "HP:0001297",  # Stroke
            "HP:0000822",  # Hypertension
        ],
        "pathway": "renal",
        "task_types": ["oligogenic"],
    },
    # ── Connective tissue ─────────────────────────────────────────────────────
    {
        "disease": "Marfan syndrome",
        "genes": ["FBN1"],
        "hpo_ids": [
            "HP:0003179",  # Protrusio acetabuli
            "HP:0000518",  # Cataract
            "HP:0001166",  # Arachnodactyly
            "HP:0002616",  # Aortic root aneurysm
            "HP:0001083",  # Ectopia lentis
        ],
        "pathway": "connective_tissue",
        "task_types": ["monogenic", "phenotype_mismatch"],
    },
    # ── Cancer predisposition (DECOYS for non-cancer tasks) ───────────────────
    {
        "disease": "Hereditary breast and ovarian cancer",
        "genes": ["BRCA1", "BRCA2"],
        "hpo_ids": [
            "HP:0003002",  # Breast carcinoma
            "HP:0100615",  # Ovarian neoplasm
            "HP:0002894",  # Pancreatic carcinoma
            "HP:0006740",  # Transitional cell carcinoma
        ],
        "pathway": "cancer",
        "task_types": [],  # Never a primary task — only used as decoy
    },
    {
        "disease": "Li-Fraumeni syndrome",
        "genes": ["TP53"],
        "hpo_ids": [
            "HP:0002671",  # Basal cell carcinoma
            "HP:0012125",  # Prostate cancer
            "HP:0001909",  # Leukaemia
            "HP:0003003",  # Colon cancer
        ],
        "pathway": "cancer",
        "task_types": [],
    },
    # ── Musculoskeletal ───────────────────────────────────────────────────────
    {
        "disease": "Duchenne muscular dystrophy",
        "genes": ["DMD"],
        "hpo_ids": [
            "HP:0003560",  # Muscular dystrophy
            "HP:0001639",  # Hypertrophic cardiomyopathy
            "HP:0001252",  # Hypotonia
            "HP:0001263",  # Global developmental delay
            "HP:0003236",  # Elevated serum creatine kinase
        ],
        "pathway": "musculoskeletal",
        "task_types": ["monogenic"],
    },
    {
        "disease": "X-linked hypophosphatemia",
        "genes": ["PHEX"],
        "hpo_ids": [
            "HP:0002748",  # Rickets
            "HP:0002652",  # Skeletal dysplasia
            "HP:0001249",  # Intellectual disability
            "HP:0000823",  # Delayed puberty
            "HP:0001155",  # Abnormal hand morphology
        ],
        "pathway": "musculoskeletal",
        "task_types": ["monogenic"],
    },
    # ── Ophthalmology ─────────────────────────────────────────────────────────
    {
        "disease": "Stargardt disease",
        "genes": ["ABCA4"],
        "hpo_ids": [
            "HP:0007663",  # Reduced visual acuity
            "HP:0000505",  # Visual impairment
            "HP:0007737",  # Bull's eye maculopathy
            "HP:0000529",  # Progressive visual loss
            "HP:0001131",  # Corneal dystrophy
        ],
        "pathway": "ophthalmology",
        "task_types": ["monogenic"],
    },
    # ── Lipid disorders ───────────────────────────────────────────────────────
    {
        "disease": "Familial hypercholesterolaemia",
        "genes": ["LDLR", "APOB", "PCSK9"],
        "hpo_ids": [
            "HP:0003124",  # Hypercholesterolaemia
            "HP:0000956",  # Acanthosis nigricans
            "HP:0001297",  # Stroke
            "HP:0001677",  # Coronary artery disease
            "HP:0000822",  # Hypertension
        ],
        "pathway": "metabolic",
        "task_types": ["monogenic", "oligogenic"],
    },
]

# Build fast lookup: disease_name → catalog entry
DISEASE_BY_NAME: Dict[str, Dict[str, Any]] = {d["disease"]: d for d in DISEASE_CATALOG}

# Lookup gene → catalog entries
GENE_TO_DISEASES: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
for _entry in DISEASE_CATALOG:
    for _gene in _entry["genes"]:
        GENE_TO_DISEASES[_gene].append(_entry)


def _slugify(text: str) -> str:
    """Make a stable node-ID-safe string from a disease name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:60]


# Module-level singleton — loaded once, reused across all episodes
_GRAPH: Optional[ClinDetectGraph] = None


def get_graph() -> ClinDetectGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ClinDetectGraph()
        _GRAPH.load()
    return _GRAPH
