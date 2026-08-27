"""W&B Offline Runs Auditor and Log Parser.

Parses local Weights & Biases offline runs, identifies completion vs. failure
states, categorizes experiments executed on the local host, and optionally
quarantines failed runs into wandb/aborted_archive/.
"""

import io
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

if isinstance(sys.stdout, io.TextIOWrapper) and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def parse_single_run(run_path: str, location: str) -> Dict[str, Any]:
    """Parses metadata, summary metrics, and protobuf records for a W&B run directory."""
    dir_name = os.path.basename(run_path)

    meta_path = os.path.join(run_path, "files", "wandb-metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(run_path, "wandb-metadata.json")

    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                meta = json.load(f)
        except Exception:
            pass

    summary_path = os.path.join(run_path, "files", "wandb-summary.json")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(run_path, "wandb-summary.json")

    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
                summary = json.load(f)
        except Exception:
            pass

    wandb_files = [
        os.path.join(run_path, f) for f in os.listdir(run_path) if f.endswith(".wandb")
    ]
    exit_code: Optional[int] = meta.get("exitcode")
    run_name: str = meta.get("program", dir_name)
    host: str = meta.get("host", "")
    history_count = 0

    if wandb_files:
        try:
            ds = DataStore()
            ds.open_for_scan(wandb_files[0])
            while True:
                data = ds.scan_data()
                if data is None:
                    break
                pb = wandb_internal_pb2.Record()  # pyright: ignore[reportAttributeAccessIssue]
                pb.ParseFromString(data)
                rtype = pb.WhichOneof("record_type")
                if rtype == "exit" and exit_code is None:
                    exit_code = pb.exit.exit_code
                elif rtype == "run":
                    if pb.run.display_name:
                        run_name = pb.run.display_name
                    if not host and pb.run.host:
                        host = pb.run.host
                elif rtype == "history":
                    history_count += 1
        except Exception:
            pass

    if not host:
        host = "CF-LILA-004-D"

    # Status classification
    is_failed = False
    if exit_code is not None:
        is_failed = exit_code != 0
    else:
        if len(summary) == 0 and history_count < 200:
            is_failed = True
        elif location == "wandb/aborted_archive/":
            is_failed = True

    timestamp = ""
    if "run-" in dir_name:
        parts = dir_name.split("run-")[1].split("-")[0]
        if len(parts) == 15 and "_" in parts:
            d_part, t_part = parts.split("_")
            timestamp = f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:]} {t_part[:2]}:{t_part[2:4]}:{t_part[4:]}"

    return {
        "id": dir_name,
        "name": run_name,
        "location": location,
        "path": run_path,
        "timestamp": timestamp,
        "host": host,
        "exit_code": exit_code,
        "is_failed": is_failed,
        "runtime": summary.get("_wandb", {}).get("runtime", 0),
        "summary": summary,
    }


def audit_wandb_directory(
    base_dir: str = "wandb",
    auto_move_failed: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Audits all runs in the workspace and optionally quarantines failed runs."""
    active_runs: List[Dict[str, Any]] = []
    archived_runs: List[Dict[str, Any]] = []

    # 1. Direct wandb runs
    direct_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d not in ["aborted_archive", "wandb"]
        and not d.startswith("sweep-")
    ]
    for d in direct_dirs:
        p = os.path.join(base_dir, d)
        run_info = parse_single_run(p, f"{base_dir}/")
        if run_info["is_failed"]:
            if auto_move_failed:
                target_dir = os.path.join(base_dir, "aborted_archive", d)
                os.makedirs(os.path.join(base_dir, "aborted_archive"), exist_ok=True)
                shutil.move(p, target_dir)
                run_info["path"] = target_dir
                run_info["location"] = f"{base_dir}/aborted_archive/"
                archived_runs.append(run_info)
            else:
                active_runs.append(run_info)
        else:
            active_runs.append(run_info)

    # 2. Nested wandb/wandb/
    nested_dir = os.path.join(base_dir, "wandb")
    if os.path.exists(nested_dir):
        for d in os.listdir(nested_dir):
            p = os.path.join(nested_dir, d)
            if os.path.isdir(p):
                active_runs.append(parse_single_run(p, f"{base_dir}/wandb/"))

    # 3. wandb/aborted_archive/
    archive_dir = os.path.join(base_dir, "aborted_archive")
    if os.path.exists(archive_dir):
        for d in os.listdir(archive_dir):
            p = os.path.join(archive_dir, d)
            if os.path.isdir(p):
                archived_runs.append(
                    parse_single_run(p, f"{base_dir}/aborted_archive/")
                )

    return {"active": active_runs, "archived": archived_runs}


def main() -> None:
    """CLI entrypoint for auditing and summarizing W&B runs."""
    results = audit_wandb_directory(base_dir="wandb", auto_move_failed=True)
    active = results["active"]
    archived = results["archived"]

    print("=" * 60)
    print(" [W&B OFFLINE RUNS AUDIT SUMMARY]")
    print("=" * 60)
    print(f"Total Audited Runs : {len(active) + len(archived)}")
    print(f"Active Unfailed    : {len(active)}")
    print(f"Archived Aborted   : {len(archived)}\n")

    # Group active runs
    sweeps = [r for r in active if "sweep" in r["id"] or "run_sweep" in r["name"]]
    dinov3 = [r for r in active if "dinov3" in r["name"].lower()]
    other = [r for r in active if r not in sweeps and r not in dinov3]

    print(f"  * YOLO HPO Sweeps     : {len(sweeps)} runs")
    print(f"  * DINOv3 Baseline/DETR: {len(dinov3)} runs")
    print(f"  * Other Experiments   : {len(other)} runs")
    print("=" * 60)


if __name__ == "__main__":
    main()
