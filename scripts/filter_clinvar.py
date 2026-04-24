"""
Filter variant_summary.txt (ClinVar) to high-confidence pathogenic variants.

Keeps GRCh38 rows where:
  - ClinicalSignificance contains Pathogenic or Likely_pathogenic
  - GeneSymbol is not empty
  - PhenotypeList is not 'not provided'
  - ReviewStatus is 'criteria provided, multiple submitters' OR 'reviewed by expert panel'
    (high-confidence only — reduces ~450K → ~60K rows, ~10MB)

Output: data/clinvar_pathogenic.tsv
Run once locally before committing the repo.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INPUT_PATH = REPO_ROOT / "data" / "variant_summary.txt"
OUTPUT_PATH = REPO_ROOT / "data" / "clinvar_pathogenic.tsv"

KEEP_COLUMNS = [
    "#AlleleID",
    "GeneSymbol",
    "ClinicalSignificance",
    "PhenotypeList",
    "PhenotypeIDS",
    "Type",
    "Name",
    "Chromosome",
    "Start",
]

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}

HIGH_CONFIDENCE_REVIEW = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}


def is_pathogenic(clnsig: str) -> bool:
    low = clnsig.lower()
    return any(t in low for t in PATHOGENIC_TERMS)


def is_high_confidence(review_status: str) -> bool:
    return review_status.strip().lower() in HIGH_CONFIDENCE_REVIEW


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_PATH} ...")
    written = 0
    seen_allele_ids: set[str] = set()

    with (
        open(INPUT_PATH, "r", encoding="utf-8", errors="replace") as fin,
        open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as fout,
    ):
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=KEEP_COLUMNS, delimiter="\t")
        writer.writeheader()

        for i, row in enumerate(reader):
            if i % 500_000 == 0 and i > 0:
                print(f"  {i:,} rows scanned, {written:,} kept ...")

            if row.get("Assembly", "") != "GRCh38":
                continue

            if not is_pathogenic(row.get("ClinicalSignificance", "")):
                continue

            gene = row.get("GeneSymbol", "").strip()
            if not gene or gene == "-":
                continue

            phenotype = row.get("PhenotypeList", "").strip()
            if not phenotype or phenotype.lower() in ("not provided", "-", ""):
                continue

            if not is_high_confidence(row.get("ReviewStatus", "")):
                continue

            # Deduplicate by AlleleID
            allele_id = row.get("#AlleleID", "")
            if allele_id in seen_allele_ids:
                continue
            seen_allele_ids.add(allele_id)

            writer.writerow({col: row.get(col, "") for col in KEEP_COLUMNS})
            written += 1

    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"\nDone.")
    print(f"  Written: {written:,} unique variants")
    print(f"  Size:    {size_mb:.1f} MB")
    print(f"  Saved:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
