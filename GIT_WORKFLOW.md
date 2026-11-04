# Git Workflow Documentation — Lab 3

**Project:** play-store (Google Play Store App Rating Prediction MLOps Pipeline)  
**Lab:** Lab 3 — Git-Based Version Control and Collaborative ML Workflows

This document records the version-control workflows demonstrated in this
repository: repository initialization, modular branching, simulated
collaborative development, merge-conflict resolution, and faulty-commit
identification with rollback.

---

## 1. Repository Initialization

The repository was initialized with baseline data exploration, validation,
and modeling workflows for the Google Play Store dataset.

| Branch | Description |
|---|---|
| `main` | Production/stable trunk |
| `feature/modular-pipeline` | Modular scripts refactoring |
| `feature/rating-threshold-tuning` | Imputation & threshold experiments |

---

## 2. Branching Strategy

Independent feature branches were created off `main` to isolate modular components:
- `src/preprocess.py` — feature cleaning (Size, Price, Reviews, Installs).
- `src/train.py` — baseline model training.
- `src/evaluate.py` — held-out test evaluation.

---

## 3. Simulated Collaborative Development & Merge Conflict Resolution

In collaborative development, two engineers made simultaneous changes to `src/preprocess.py` handling missing Rating values:
- Engineer 1: Imputed missing Rating using median strategy.
- Engineer 2: Imputed missing Rating using mean strategy.

Conflict resolution strategy:
Rather than imputing artificial ratings, apps missing ratings were removed from training to avoid introducing bias into the continuous rating and churn proxy target, preserving clean ground-truth data.

---

## 4. Faulty Commit Identification & Rollback

A faulty commit introducing zero division during scaling was identified, isolated, and reverted using `git revert`, restoring pipeline integrity.
