"""
Production Data Drift Detection Engine
Calculates Population Stability Index (PSI) and statistical tests
(Kolmogorov-Smirnov for numerical, Chi-Square for categorical)
comparing production inference inputs against the baseline training distribution.
"""

import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats


def calculate_numerical_psi(
    baseline: pd.Series,
    production: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, float, str]:
    """
    Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov test for a numerical feature.
    Returns: (psi_value, ks_pvalue, drift_status)
    """
    base_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    prod_clean = pd.to_numeric(production, errors="coerce").dropna()

    if len(base_clean) < 5 or len(prod_clean) < 5:
        return 0.0, 1.0, "NO SIGNIFICANT DRIFT"

    # KS 2-sample test
    ks_stat, ks_pvalue = stats.ks_2samp(base_clean, prod_clean)

    # Quantile bins based on baseline
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(base_clean, quantiles * 100)
    bin_edges = np.unique(bin_edges)

    # Edge case: if values are uniform or identical
    if len(bin_edges) < 2:
        bin_edges = np.array([-np.inf, np.inf])
    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    # Assign counts to bins
    base_counts, _ = np.histogram(base_clean, bins=bin_edges)
    prod_counts, _ = np.histogram(prod_clean, bins=bin_edges)

    # Calculate proportions with smoothing epsilon
    base_prop = (base_counts + epsilon) / (len(base_clean) + epsilon * len(base_counts))
    prod_prop = (prod_counts + epsilon) / (len(prod_clean) + epsilon * len(prod_counts))

    # PSI formula: sum((prod - base) * ln(prod / base))
    psi_val = np.sum((prod_prop - base_prop) * np.log(prod_prop / base_prop))
    psi_val = float(np.nan_to_num(psi_val, nan=0.0, posinf=1.0, neginf=0.0))

    if psi_val >= 0.25:
        status = "HIGH DRIFT"
    elif psi_val >= 0.10:
        status = "MODERATE DRIFT"
    else:
        status = "NO SIGNIFICANT DRIFT"

    return round(psi_val, 4), round(float(ks_pvalue), 4), status


def calculate_categorical_psi(
    baseline: pd.Series,
    production: pd.Series,
    epsilon: float = 1e-4
) -> Tuple[float, float, str]:
    """
    Calculates Population Stability Index (PSI) and Chi-Square goodness-of-fit for a categorical feature.
    Returns: (psi_value, chi2_pvalue, drift_status)
    """
    base_clean = baseline.astype(str).str.strip().dropna()
    prod_clean = production.astype(str).str.strip().dropna()

    if len(base_clean) < 5 or len(prod_clean) < 5:
        return 0.0, 1.0, "NO SIGNIFICANT DRIFT"

    all_categories = sorted(list(set(base_clean.unique()).union(set(prod_clean.unique()))))

    base_freq = base_clean.value_counts()
    prod_freq = prod_clean.value_counts()

    base_counts = np.array([base_freq.get(c, 0) for c in all_categories])
    prod_counts = np.array([prod_freq.get(c, 0) for c in all_categories])

    # Proportions with smoothing
    base_prop = (base_counts + epsilon) / (len(base_clean) + epsilon * len(all_categories))
    prod_prop = (prod_counts + epsilon) / (len(prod_clean) + epsilon * len(all_categories))

    psi_val = np.sum((prod_prop - base_prop) * np.log(prod_prop / base_prop))
    psi_val = float(np.nan_to_num(psi_val, nan=0.0, posinf=1.0, neginf=0.0))

    # Chi-square test
    try:
        expected_counts = base_prop * len(prod_clean)
        chi2_stat, chi2_pvalue = stats.chisquare(f_obs=prod_counts + epsilon, f_exp=expected_counts + epsilon)
        chi2_pvalue = float(chi2_pvalue)
    except Exception:
        chi2_pvalue = 1.0

    if psi_val >= 0.25:
        status = "HIGH DRIFT"
    elif psi_val >= 0.10:
        status = "MODERATE DRIFT"
    else:
        status = "NO SIGNIFICANT DRIFT"

    return round(psi_val, 4), round(chi2_pvalue, 4), status


class DataDriftDetector:
    """
    Evaluates production data against training baseline, logs alerts, and exports CSV reports.
    """
    def __init__(
        self,
        baseline_path: str = "data/processed/processed_googleplaystore.csv",
        reports_dir: str = "reports",
        alerts_log_path: str = "logs/drift_alerts.log"
    ):
        self.baseline_path = baseline_path
        self.reports_dir = reports_dir
        self.alerts_log_path = alerts_log_path
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.alerts_log_path), exist_ok=True)
        self.baseline_df = self._load_baseline()

    def _load_baseline(self) -> pd.DataFrame:
        if not os.path.exists(self.baseline_path):
            raise FileNotFoundError(f"Baseline training data not found at: {self.baseline_path}")
        df = pd.read_csv(self.baseline_path)
        print(f"[INFO] Loaded baseline dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def compute_drift(self, production_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes feature-by-feature drift and overall dataset drift level.
        """
        numerical_features = [
            "Reviews", "Size", "Installs", "Price",
            "Current Ver", "Android Ver"
        ]
        categorical_features = [
            "Category", "Type", "Content Rating", "Genres"
        ]

        report_rows = []
        high_drift_features = []
        moderate_drift_features = []

        # Analyze numerical features
        for feat in numerical_features:
            if feat in production_df.columns and feat in self.baseline_df.columns:
                psi, p_val, status = calculate_numerical_psi(
                    self.baseline_df[feat], production_df[feat]
                )
                report_rows.append({
                    "Feature": feat,
                    "Type": "Numerical",
                    "PSI": psi,
                    "Statistical_P_Value": p_val,
                    "Statistical_Test": "Kolmogorov-Smirnov",
                    "Drift_Status": status
                })
                if status == "HIGH DRIFT":
                    high_drift_features.append(feat)
                elif status == "MODERATE DRIFT":
                    moderate_drift_features.append(feat)

        # Analyze categorical features
        for feat in categorical_features:
            if feat in production_df.columns and feat in self.baseline_df.columns:
                psi, p_val, status = calculate_categorical_psi(
                    self.baseline_df[feat], production_df[feat]
                )
                report_rows.append({
                    "Feature": feat,
                    "Type": "Categorical",
                    "PSI": psi,
                    "Statistical_P_Value": p_val,
                    "Statistical_Test": "Chi-Square",
                    "Drift_Status": status
                })
                if status == "HIGH DRIFT":
                    high_drift_features.append(feat)
                elif status == "MODERATE DRIFT":
                    moderate_drift_features.append(feat)

        # Overall severity
        if len(high_drift_features) > 0:
            overall_severity = "HIGH DRIFT"
        elif len(moderate_drift_features) > 0:
            overall_severity = "MODERATE DRIFT"
        else:
            overall_severity = "NO SIGNIFICANT DRIFT"

        report_df = pd.DataFrame(report_rows)

        # Save timestamped CSV report
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"drift_report_{timestamp_str}.csv"
        report_filepath = os.path.join(self.reports_dir, report_filename)
        report_df.to_csv(report_filepath, index=False)

        # Save summary log to logs/drift_alerts.log
        log_entry = (
            f"[{datetime.datetime.now().isoformat()}] SEVERITY: {overall_severity} | "
            f"Production Samples: {len(production_df)} | "
            f"High Drift Features: {high_drift_features} | "
            f"Report: {report_filepath}\n"
        )
        with open(self.alerts_log_path, "a") as f:
            f.write(log_entry)

        return {
            "overall_severity": overall_severity,
            "timestamp": datetime.datetime.now().isoformat(),
            "production_sample_size": len(production_df),
            "high_drift_features": high_drift_features,
            "moderate_drift_features": moderate_drift_features,
            "report_filepath": report_filepath,
            "report_summary": report_rows
        }
