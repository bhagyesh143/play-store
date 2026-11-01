"""
Automated Production Drift Monitoring Pipeline
Periodically reads inference audit logs, measures Population Stability Index (PSI)
against baseline training data, saves CSV reports, and sends email alerts upon HIGH drift.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime

from monitoring.drift_detector import DataDriftDetector
from monitoring.alert_service import email_alert_service
from api.preprocessing import (
    clean_reviews, clean_size, clean_installs,
    clean_price, clean_version, clean_android_version,
    parse_date_features
)


class AutomaticDriftMonitor:
    def __init__(
        self,
        log_path: str = "logs/inference/prediction_logs.jsonl",
        baseline_path: str = "data/processed/processed_googleplaystore.csv",
        reports_dir: str = "reports",
        min_samples: int = 5
    ):
        self.log_path = log_path
        self.min_samples = min_samples
        self.last_processed_line = 0
        self.last_log_file_size = 0
        self.detector = DataDriftDetector(
            baseline_path=baseline_path,
            reports_dir=reports_dir
        )

    def load_production_logs(self) -> pd.DataFrame:
        """
        Reads inference logs and transforms them into a clean feature DataFrame.
        """
        if not os.path.exists(self.log_path):
            print(f"[INFO] No inference log found yet at {self.log_path}. Waiting for API traffic...")
            return pd.DataFrame()

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("status") == "SUCCESS":
                        feat = data.get("input_features", {})
                        records.append(feat)
                except Exception:
                    continue

        if not records:
            return pd.DataFrame()

        # Transform raw inference inputs to match baseline features
        rows = []
        for r in records:
            y, m, d = parse_date_features(r.get("Last Updated", r.get("Last_Updated", None)))
            row = {
                "App": str(r.get("App", "Unknown")),
                "Category": str(r.get("Category", "FAMILY")),
                "Reviews": clean_reviews(r.get("Reviews", 100)),
                "Size": clean_size(r.get("Size", 15.0)),
                "Installs": clean_installs(r.get("Installs", 10000)),
                "Type": str(r.get("Type", "Free")),
                "Price": clean_price(r.get("Price", 0.0)),
                "Content Rating": str(r.get("Content Rating", r.get("Content_Rating", "Everyone"))),
                "Genres": str(r.get("Genres", "Tools")),
                "Current Ver": clean_version(r.get("Current Ver", r.get("Current_Ver", "1.0"))),
                "Android Ver": clean_android_version(r.get("Android Ver", r.get("Android_Ver", "4.0"))),
                "Updated_Year": y,
                "Updated_Month": m,
                "Updated_Day": d
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def run_check(self) -> bool:
        """
        Executes one monitoring cycle. Returns True if check executed, False if skipped.
        """
        print("\n" + "=" * 65)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] RUNNING AUTOMATED DRIFT CHECK")
        print("=" * 65)

        if not os.path.exists(self.log_path):
            print(f"[INFO] Waiting for production traffic: {self.log_path} does not exist yet.")
            return False

        current_size = os.path.getsize(self.log_path)
        if current_size == 0:
            print("[INFO] Inference log file is empty. Send predictions to the API first.")
            return False

        prod_df = self.load_production_logs()

        if len(prod_df) < self.min_samples:
            print(
                f"[INFO] Insufficient production logs ({len(prod_df)}/{self.min_samples} required). "
                f"Statistical drift check will activate after {self.min_samples} inference requests."
            )
            return False

        # Run drift computation
        results = self.detector.compute_drift(prod_df)

        print("\n" + "-" * 65)
        print("FEATURE-WISE DRIFT ANALYSIS")
        print("-" * 65)
        print(f"{'Feature':<16} {'Type':<12} {'PSI':<8} {'P-Value':<10} {'Status':<16}")
        print("-" * 65)
        for item in results["report_summary"]:
            print(
                f"{item['Feature']:<16} {item['Type']:<12} {item['PSI']:<8.4f} "
                f"{item['Statistical_P_Value']:<10.4f} {item['Drift_Status']:<16}"
            )
        print("-" * 65)

        severity = results["overall_severity"]
        badge = "🟢" if severity == "NO SIGNIFICANT DRIFT" else ("🟡" if severity == "MODERATE DRIFT" else "🔴")
        print(f"\n{badge} OVERALL DATASET DRIFT STATUS: {severity}")
        print(f"📄 Detailed CSV Report Saved: {results['report_filepath']}")
        print(f"📝 Alert Log Updated: {self.detector.alerts_log_path}")

        # Trigger email alerts on HIGH DRIFT
        if severity == "HIGH DRIFT":
            print(f"🚨 High drift detected in features: {results['high_drift_features']}")
            email_alert_service.send_drift_alert(results)

        return True


def main():
    parser = argparse.ArgumentParser(description="Automated Production Drift Monitor")
    parser.add_argument("--once", action="store_true", help="Run drift check once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Continuous check interval in seconds (default: 60)")
    args = parser.parse_args()

    monitor = AutomaticDriftMonitor()

    if args.once:
        monitor.run_check()
    else:
        print(f"[INFO] Starting continuous drift monitor (Polling interval: {args.interval}s)")
        print("[INFO] Press Ctrl+C to terminate the monitor.")
        try:
            while True:
                monitor.run_check()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[INFO] Automated drift monitor stopped by user.")


if __name__ == "__main__":
    main()
