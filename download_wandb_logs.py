"""
W&B Online Project Downloader.

Downloads all experiment summaries, configurations, and execution log files
from the Weights & Biases cloud workspace into a local directory for offline analysis.
"""

import json
import os
import wandb


def download_wandb_project(
    entity: str = "brezo-boku-vienna",
    project: str = "_baseline",
    output_dir: str = "wandb_downloaded_logs",
) -> str:
    """Downloads all run metrics, configs, and logs from W&B project."""
    api = wandb.Api()
    target_dir = os.path.abspath(output_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Connecting to W&B project: {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs online. Downloading into: {target_dir}\n")

    downloaded_count = 0
    for idx, r in enumerate(runs, 1):
        safe_name = "".join(
            [c if c.isalnum() or c in ("-", "_") else "_" for c in r.name]
        )
        run_dir = os.path.join(target_dir, f"{r.id}_{safe_name}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(r.summary._json_dict, f, indent=2)

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(r.config, f, indent=2)

        files = list(r.files())
        log_files = [
            f
            for f in files
            if f.name.endswith((".log", ".json", ".yaml", ".txt", ".csv"))
        ]
        for f in log_files:
            try:
                f.download(root=run_dir, replace=True)
            except Exception as e:
                print(f"Skipped {f.name} for {r.id}: {e}")

        downloaded_count += 1
        if idx % 5 == 0 or idx == len(runs):
            print(f"Progress: {idx}/{len(runs)} runs downloaded.")

    print(
        f"\nSuccessfully downloaded logs for all {downloaded_count} runs to: {target_dir}"
    )
    return target_dir


if __name__ == "__main__":
    download_wandb_project()
