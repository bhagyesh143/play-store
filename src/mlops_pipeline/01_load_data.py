import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
import pandas as pd
import yaml


def calculate_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_data():
    print("[INFO] Starting Data Loading & Ingestion Pipeline...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    new_raw_path = config["data"].get("new_raw_path", "data/raw/new_googleplaystore.csv")
    archive_dir = config["data"].get("archive_dir", "data/raw/archive")
    lineage_path = config["data"].get("lineage_path", "data/raw/archive/dataset_lineage.json")

    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"[ERROR] Raw dataset not found at: {raw_path}")
        return False

    current_df = pd.read_csv(raw_path)
    current_hash = calculate_file_hash(raw_path)

    # Initialize lineage if missing
    if not os.path.exists(lineage_path):
        lineage_data = {
            "dataset_name": "Google Play Store Apps",
            "base_file": raw_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": [
                {
                    "version": 1,
                    "filename": "googleplaystore_v01.csv",
                    "hash": current_hash,
                    "rows": int(len(current_df)),
                    "columns": int(current_df.shape[1]),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "initial_baseline"
                }
            ],
            "ingested_hashes": [current_hash]
        }
        # Save baseline snapshot
        shutil.copyfile(raw_path, os.path.join(archive_dir, "googleplaystore_v01.csv"))
        with open(lineage_path, "w", encoding="utf-8") as f:
            json.dump(lineage_data, f, indent=4)
    else:
        with open(lineage_path, "r", encoding="utf-8") as f:
            lineage_data = json.load(f)

    data_status = "baseline_loaded"
    new_batch_ingested = False

    # Check incoming batch
    if os.path.exists(new_raw_path):
        print(f"[INFO] Detected incoming batch at {new_raw_path}")
        batch_hash = calculate_file_hash(new_raw_path)

        if batch_hash in lineage_data.get("ingested_hashes", []):
            print(f"[INFO] Batch hash {batch_hash[:12]} already present in lineage. Skipping duplicate ingestion.")
            data_status = "already_ingested"
        else:
            print(f"[INFO] Validating and ingesting new batch (SHA256: {batch_hash[:12]})...")
            batch_df = pd.read_csv(new_raw_path)

            # Check schema consistency
            if list(batch_df.columns) != list(current_df.columns):
                print(f"[ERROR] Incoming batch columns do not match raw dataset schema!")
                return False

            next_ver = len(lineage_data["versions"]) + 1
            archive_snapshot = f"googleplaystore_v{next_ver:02d}.csv"
            shutil.copyfile(raw_path, os.path.join(archive_dir, archive_snapshot))

            # Deduplicate by App name if present
            id_col = config["data"].get("id_column", "App")
            if id_col in current_df.columns:
                existing_ids = set(current_df[id_col].dropna().astype(str))
                batch_df = batch_df[~batch_df[id_col].astype(str).isin(existing_ids)]

            combined_df = pd.concat([current_df, batch_df], ignore_index=True)
            combined_df.to_csv(raw_path, index=False)
            current_df = combined_df

            lineage_data["versions"].append({
                "version": next_ver,
                "filename": archive_snapshot,
                "hash": batch_hash,
                "rows": int(len(current_df)),
                "new_rows_added": int(len(batch_df)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": os.path.basename(new_raw_path)
            })
            lineage_data["ingested_hashes"].append(batch_hash)

            with open(lineage_path, "w", encoding="utf-8") as f:
                json.dump(lineage_data, f, indent=4)

            data_status = "ingested"
            new_batch_ingested = True
            print(f"[SUCCESS] Ingested {len(batch_df)} rows. Dataset expanded to {len(current_df)} rows.")

    report = {
        "stage": "ML Pipeline - Data Loading",
        "status": data_status,
        "new_data_ingested": new_batch_ingested,
        "source_file": raw_path,
        "rows": int(current_df.shape[0]),
        "columns": int(current_df.shape[1]),
        "column_names": current_df.columns.tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    report_path = "artifacts/data_load_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Data loading completed. Status: {data_status}. Report: {report_path}")
    return True


if __name__ == "__main__":
    passed = load_data()
    sys.exit(0 if passed else 1)
