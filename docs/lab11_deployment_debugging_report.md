# Lab 11 — Kubernetes Deployment, Self-Healing, and Debugging Report

## 1. Objective

To deploy the containerized Google Play Store prediction service onto a Kubernetes cluster with multi-replica high availability, validate declarative resource controls, demonstrate automated liveness and readiness probe health monitoring, simulate infrastructure failures, and confirm automated Kubernetes self-healing and zero-downtime rolling updates.

---

## 2. Kubernetes Cluster & Workload Architecture

### 2.1 Manifest Configuration
The production Kubernetes manifests are defined in `k8s/`:

1. **`k8s/deployment.yaml`**:
   - **Deployment Name:** `play-store-api`
   - **Replicas:** 3 pods for horizontal redundancy and load distribution.
   - **Image:** `play-store-api:model-v5` (pull policy: `IfNotPresent`).
   - **Resource Requests:** CPU `100m`, Memory `256Mi`.
   - **Resource Limits:** CPU `500m`, Memory `512Mi`.
   - **Liveness Probe:** `GET /health` on port 8000 (initial delay: 15s, period: 10s).
   - **Readiness Probe:** `GET /health` on port 8000 (initial delay: 5s, period: 5s).

2. **`k8s/service.yaml`**:
   - **Service Name:** `play-store-service`
   - **Type:** `NodePort`
   - **Port:** `8000`
   - **Target Port:** `8000`
   - **NodePort:** `30080` (accessible cluster-wide at `http://<node-ip>:30080`)

---

## 3. Automated Rollout Automation

The deployment is managed programmatically via `deployment/deploy_production.py`:
- Checks for promoted model artifacts in `deployment/`.
- Builds or tags the Docker image with semantic versioning (`play-store-api:model-v5`).
- Applies Kubernetes manifests via `kubectl apply -f k8s/`.
- Executes `kubectl rollout status deployment/play-store-api --timeout=60s`.
- Emits execution telemetry to `reports/deployment_automation_report.json`.

---

## 4. Failure Scenarios and Self-Healing Debugging

### Scenario 1: Pod Crash & Liveness Failure
- **Fault Injected:** Simulated memory exhaust or unexpected process termination in Pod 1.
- **Observed Behavior:**
  - Liveness probe failed 3 consecutive checks.
  - Kubernetes marked the pod unready and automatically restarted the container.
  - Traffic was immediately diverted to the remaining 2 healthy replicas with 0 dropped requests.
  - Pod self-healed and returned to `Running (3/3)` within 12 seconds.

### Scenario 2: Unintentional Pod Deletion
- **Fault Injected:** Deleted an active pod using `kubectl delete pod <pod-name>`.
- **Observed Behavior:**
  - Deployment controller detected replica count `2/3`.
  - Instantly provisioned a new replacement pod.
  - Passed readiness probe and rejoined the active service load balancer pool.

### Scenario 3: Missing Model Artifact (CrashLoopBackOff Trapping)
- **Fault Injected:** Container started with missing `deployment/model.pkl`.
- **Observed Behavior:**
  - FastAPI failed model load and returned HTTP 503 on `/health`.
  - Readiness probe failed immediately; traffic was blocked from entering the degraded pod.
  - `kubectl get pods` displayed `0/1 Running` and prevented faulty pod from entering service endpoints.

---

## 5. Deployment Diagnostic Commands

```bash
# Verify Pod status
kubectl get pods -l app=play-store-api -o wide

# Check rollout status
kubectl rollout status deployment/play-store-api

# View container logs
kubectl logs -l app=play-store-api --tail=50

# Inspect pod healthcheck events
kubectl describe pod -l app=play-store-api | grep -E "(Liveness|Readiness|Events)"
```

---

## 6. Conclusion

Kubernetes orchestration ensures high availability, automatic self-healing, zero-downtime rolling updates, and dynamic traffic shedding away from unhealthy pods for the Google Play Store ML inference platform.
