# LAB 8: Continuous Integration & Quality Gate Report

## 1. Objective

To automate validation, testing, and automated quality gating of the Google Play Store MLOps project using GitHub Actions. The goal is to establish automated guardrails that prevent corrupted datasets or sub-par models (failing the production quality threshold) from progressing into deployment.

---

## 2. Continuous Integration Workflow

The GitHub Actions workflow is defined in `.github/workflows/lab8_ci.yml`. It runs on both `push` and `pull_request` events targeting `main`.

### Workflow Stages:
1. **Repository Checkout:** Clones the codebase and verifies git integrity.
2. **Environment Setup:** Configures Python runtime (3.11 / 3.12) and sets up dependency caching.
3. **Dependency Installation:** Installs `requirements.txt` including Scikit-Learn, Pandas, Pandera, and MLflow.
4. **File Integrity Verification:** Validates that key configuration files (`config.yaml`), training pipelines, and dataset directories exist.
5. **Data Schema Validation:** Executes `src/mlops_pipeline/02_validate_data.py` to ensure raw data satisfies Pandera schema rules.
6. **Quality Gate Verification:** Executes `src/mlops_pipeline/07_quality_gate.py` to evaluate the candidate model metrics against configured SLAs.

---

## 3. Quality Gate Specification

The quality gate configuration is maintained centrally in `config.yaml`:

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

### SLA Rationale:
In Google Play Store rating churn and low-rating risk prediction, **Recall** is prioritized to ensure that high-risk applications (apps likely to fall into the dissatisfied/churn tier, $Rating < 4.0$) are correctly identified for remediation. A candidate model must achieve a test recall score of at least 70.0% (`recall >= 0.70`) to pass.

---

## 4. Execution Results & Evidence

- **Workflow Status:** Passed successfully.
- **Data Validation Output:** `artifacts/baseline_validation.csv` generated with 0 critical schema violations.
- **Candidate Evaluation:**
  - Recall: `0.7420` (exceeds threshold of `0.7000`)
  - ROC-AUC: `0.7915`
  - F1 Score: `0.6884`
- **Quality Gate Output:** Recorded in `reports/quality_gate_report.json`:
  ```json
  {
    "metric": "recall",
    "threshold": 0.70,
    "actual_value": 0.7420,
    "passed": true,
    "decision": "PASSED"
  }
  ```

---

## 5. Conclusion

Automated CI and quality gates prevent regressions, validate schemas before resource-intensive training, and ensure that only compliant, production-grade candidate models qualify for model registry promotion.
