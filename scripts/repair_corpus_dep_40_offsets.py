#!/usr/bin/env python3
"""Corpus dependency head offset repair script for VNLP chunk_size=40.

Scans canonical thesis pickle.xz files, computes or applies idempotent shifts,
strictly excludes test_cache (Rule I), and generates JSON audit reports.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import tarfile
from typing import Dict, List, Optional

# Ensure parent directory is on sys.path for vnlp_colab imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from vnlp_colab.dep.dep_migration import adjust_dataframe_dep_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dep40_repair")

CANONICAL_CACHE_BOOKS = {
    "AscannerDarkly",
    "I_Robot",
    "TheLight",
    "TheLonelyPlanet",
    "TheScarletPlague",
}


def discover_target_files(workspace_root: Path) -> List[Path]:
    """Find all canonical target .pickle.xz files, strictly excluding test_cache."""
    data_dir = workspace_root / "data"
    targets: List[Path] = []

    # 1. data/alignment/
    for p in sorted((data_dir / "alignment").glob("**/*.pickle.xz")):
        targets.append(p)

    # 2. data/indigenous/
    for p in sorted((data_dir / "indigenous").glob("**/*.pickle.xz")):
        targets.append(p)

    # 3. data/Neologism_Dictionary_Annotation_Cache/ (strictly canonical books, exclude test_cache)
    cache_base = data_dir / "Neologism_Dictionary_Annotation_Cache"
    for book in sorted(CANONICAL_CACHE_BOOKS):
        book_dir = cache_base / book
        if book_dir.exists():
            for p in sorted(book_dir.glob("*_updated.pickle.xz")):
                if "test_cache" not in str(p):
                    targets.append(p)

    # 4. data/processed/turkish/SPACY_ENRICHED_VNLP/
    proc_dir = data_dir / "processed" / "turkish" / "SPACY_ENRICHED_VNLP"
    for p in sorted(proc_dir.glob("**/*.pickle.xz")):
        targets.append(p)

    # Final safety filter: enforce Rule I
    safe_targets = [p for p in targets if "test_cache" not in str(p.relative_to(workspace_root))]
    return sorted(list(dict.fromkeys(safe_targets)))


def create_pre_op_backup(files: List[Path], workspace_root: Path) -> Path:
    """Create a tar.xz pre-op archive of all files about to be modified."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = workspace_root / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"pre_dep40_migration_{ts}.tar.xz"

    logger.info("Creating pre-op backup archive at %s (%d files)...", backup_path, len(files))
    with tarfile.open(backup_path, "w:xz") as tar:
        for f in files:
            arcname = str(f.relative_to(workspace_root))
            tar.add(f, arcname=arcname)
    return backup_path


def run_migration(
    workspace_root: Path, apply: bool = False, report_out: Optional[Path] = None
) -> Dict[str, object]:
    """Execute dry-run or apply migration across discovered corpus files."""
    targets = discover_target_files(workspace_root)
    logger.info("Discovered %d canonical target files across 4 directories.", len(targets))

    file_results: List[Dict[str, object]] = []
    files_to_modify: List[Path] = []
    dfs_to_save: Dict[Path, pd.DataFrame] = {}

    summary = {
        "total_files_scanned": len(targets),
        "files_modified": 0,
        "total_rows_inspected": 0,
        "total_multi_chunk_rows": 0,
        "total_rows_modified": 0,
        "total_heads_shifted": 0,
        "total_dep_head_oob": 0,
        "total_already_global_chunks": 0,
        "total_all_root_chunks": 0,
    }

    for p in targets:
        rel_p = str(p.relative_to(workspace_root))
        try:
            df = pd.read_pickle(p, compression="xz")
            df_adj, tel = adjust_dataframe_dep_chunks(df)

            summary["total_rows_inspected"] += tel["rows_inspected"]
            summary["total_multi_chunk_rows"] += tel["multi_chunk_rows"]
            summary["total_rows_modified"] += tel["rows_modified"]
            summary["total_heads_shifted"] += tel["heads_shifted"]
            summary["total_dep_head_oob"] += tel["dep_head_oob"]
            summary["total_already_global_chunks"] += tel["already_global_chunks"]
            summary["total_all_root_chunks"] += tel["all_root_chunks"]

            if tel["rows_modified"] > 0:
                summary["files_modified"] += 1
                files_to_modify.append(p)
                dfs_to_save[p] = df_adj
                status = "MODIFIED"
            else:
                status = "UNCHANGED"

            file_results.append({"path": rel_p, "status": status, "telemetry": tel})
        except Exception as e:
            logger.error("Error processing %s: %s", rel_p, e)
            file_results.append({"path": rel_p, "status": "ERROR", "error": str(e)})

    backup_path_str: Optional[str] = None
    if apply and files_to_modify:
        backup_p = create_pre_op_backup(files_to_modify, workspace_root)
        backup_path_str = str(backup_p)

        logger.info("Applying changes to %d files...", len(files_to_modify))
        for p, df_new in dfs_to_save.items():
            df_new.to_pickle(p, compression="xz")
            logger.info("Saved repaired pickle: %s", p.relative_to(workspace_root))

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "apply" if apply else "dry-run",
        "workspace_root": str(workspace_root),
        "backup_path": backup_path_str,
        "summary": summary,
        "file_reports": file_results,
    }

    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON report to %s", report_out)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair corpus dep chunk offsets")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writes")
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place with backup")
    parser.add_argument("--report", type=str, default=None, help="Output path for JSON report")
    parser.add_argument(
        "--workspace-root", type=str, default="/home/ben/Documents/thesis", help="Workspace root"
    )
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    ws_root = Path(args.workspace_root).resolve()
    rep_out = Path(args.report).resolve() if args.report else None

    report = run_migration(ws_root, apply=apply, report_out=rep_out)
    logger.info("Finished %s. Summary: %s", report["mode"], json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
