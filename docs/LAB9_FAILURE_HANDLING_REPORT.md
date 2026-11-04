# LAB 9: Failure Handling, Fault Injection, and Robustness Testing

## 1. Objective

To evaluate and guarantee the robustness, error handling, and fault tolerance of the Google Play Store MLOps pipeline under data corruption and adverse runtime conditions. The module injects artificial faults into the dataset, verifies that validation layers intercept corrupted payloads immediately, and ensures detailed diagnostics are emitted without crashing pipeline infrastructure.

---

## 2. Failure Scenarios Injected

The failure injection harness (`src/mlops_pipeline/10_failure_injection.py`) systematically simulates five real-world data corruption scenarios:

| # | Failure Scenario | Injection Method | Detection Mechanism | Expected & Verified Outcome |
|:---:|---|---|---|---|
| **1** | **Missing Raw Dataset** | Simulates absent `googleplaystore.csv` path | File existence verification in `01_load_data.py` | Pipeline halts with `FileNotFoundError`, records failure status in run log. |
| **2** | **Missing Target Column** | Drops `Rating` / `Rating_Tier` from input dataframe | Schema validation in `02_validate_data.py` | Pandera flags missing required column; raises `SchemaError`. |
| **3** | **Missing Values in Critical Features** | Injects `NaN` values into `Category`, `Reviews`, `Installs`, or `Price` | Pandera `nullable=False` check | Schema validation intercepts unhandled nulls and reports row indices. |
| **4** | **Corrupted Target Values** | Injects out-of-boundary values (e.g., `Rating = 19.0` or `Rating = -2.5`) | Boundary check: $1.0 \le Rating \le 5.0$ | Validation flags boundary violations; logs invalid records to CSV. |
| **5** | **Duplicate App Records** | Duplicates existing `App` entries with contradictory metadata | Identity check on unique keys | Flags duplicate collisions; rejects batch ingestion. |

---

## 3. Failure Detection Reports Generated

Running `python src/mlops_pipeline/10_failure_injection.py` generates two structured diagnostic artifacts:

1. **`artifacts/failure_simulation_report.json`**:
   - Comprehensive summary of simulated scenarios, interception status, error codes, and detection latencies.
2. **`artifacts/corrupted_simulation_errors.csv`**:
   - Detailed per-row violation logs detailing the exact corrupted column, invalid value, and violated schema constraint.

---

## 4. GitHub Actions CI Automation

The failure testing workflow is automated via `.github/workflows/lab9_failure_tests.yml`:
- Runs tests on automated triggers and pull requests.
- Executes `10_failure_injection.py`.
- Verifies that `failure_simulation_report.json` and `corrupted_simulation_errors.csv` are produced.
- Verifies that all 5 failure scenarios are successfully intercepted and trapped.

---

## 5. Recovery Procedures

When any of the five failure scenarios trigger in production:
1. **Missing Data:** Check upstream data sync jobs and verify S3/GCS or disk paths.
2. **Missing Schema Columns:** Compare schema drift report against upstream API contracts; reject malformed ingestion batches.
3. **Invalid Boundaries / NaNs:** Inspect upstream scraping or telemetry logs; quarantine offending batch into `data/quarantine/`.
4. **Duplicate Batches:** Lineage tracking in `dataset_lineage.json` automatically discards identical SHA-256 batches.

---

## 6. Conclusion

The fault injection suite confirms that the Google Play Store MLOps pipeline fails safely and deterministically when supplied with malformed, corrupt, or incomplete data, preventing model corruption and silent inference degradation.
