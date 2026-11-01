# Google Play Store Rating Prediction MLOps - Viva Questions & Answers

This document provides comprehensive technical questions and expert-level answers covering Machine Learning, MLOps, API Engineering, Docker, Data Drift, and Production Monitoring.

---

### Q1: What is the business and technical objective of this project?
**Answer:**  
The objective is to develop a production-ready, industry-grade MLOps system that predicts Google Play Store mobile application ratings ($1.0$ to $5.0$) using application metadata (Category, Reviews, Size, Installs, Price, Content Rating, Genres, and Release Versions). Beyond training a model, the project operationalizes the lifecycle through MLflow experiment tracking, a FastAPI REST microservice, Docker containerization, real-time inference logging, Population Stability Index (PSI) data drift monitoring, and automated email alerts.

---

### Q2: Why is this treated as a Regression problem rather than Classification?
**Answer:**  
App ratings on the Google Play Store are continuous numerical values ranging from $1.0$ to $5.0$ stars (e.g. 4.2, 3.8, 4.6). While some academic labs simplify the problem into a binary classification proxy (e.g., rating $\ge 4.0$ vs $< 4.0$), real-world application stores and developers require precise rating estimations. Treating it as a continuous regression problem allows us to measure exact prediction errors using MAE and RMSE.

---

### Q3: What regression algorithms did you evaluate, and which one won? Why?
**Answer:**  
We evaluated four algorithms:
1. **Linear Regression (Baseline OLS):** Fast, but assumes linear relationships; achieved MAE: 0.3494, RMSE: 0.4980, $R^2$: 0.0573.
2. **Decision Tree Regressor:** Prone to variance and overfitting on split thresholds; achieved MAE: 0.3391, RMSE: 0.5165, $R^2$: -0.0141.
3. **Random Forest Regressor:** An ensemble of 200 bagging decision trees; achieved the best balance with **MAE: 0.3128, RMSE: 0.4663, $R^2$: 0.1734**, and an accuracy of **92.57%**.
4. **Gradient Boosting Regressor:** Sequential boosting trees; achieved MAE: 0.3221, RMSE: 0.4733, $R^2$: 0.1485.

**Random Forest Regressor** was selected as champion because its ensemble bagging technique effectively smooths out variance from noisy review counts and skewed install magnitudes.

---

### Q4: Explain the difference between MAE, MSE, and RMSE in regression.
**Answer:**  
- **Mean Absolute Error (MAE):** $\frac{1}{n} \sum |y - \hat{y}|$. Measures the average absolute error in the same units as the target variable (stars). It is linear and robust to outliers. Our model achieves an MAE of $\approx 0.31$ stars.
- **Mean Squared Error (MSE):** $\frac{1}{n} \sum (y - \hat{y})^2$. Penalizes large errors heavily by squaring them, but is expressed in squared units ($\text{stars}^2$).
- **Root Mean Squared Error (RMSE):** $\sqrt{\text{MSE}}$. Restores the error metric back to the original units (stars) while retaining sensitivity to large outlier prediction errors. Our champion model achieved the lowest RMSE of $0.4663$.

---

### Q5: How did your preprocessing pipeline prevent data leakage?
**Answer:**  
We used Scikit-Learn's `ColumnTransformer` combined with a `Pipeline`:
1. The dataset was split into train ($80\%$) and test ($20\%$) sets **before** computing feature scaling parameters.
2. The `StandardScaler` learned means and standard deviations strictly on the training partition ($X_{\text{train}}$) and transformed $X_{\text{test}}$ using those fitted values.
3. The `OneHotEncoder` was configured with `handle_unknown="ignore"` so that new or unseen categories in production do not crash the pipeline.
4. The entire preprocessor and estimator were serialized together into a single file (`best_google_play_rating_model.pkl`), ensuring exact consistency between training and live inference.

---

### Q6: What role does MLflow play in this architecture?
**Answer:**  
MLflow provides full lifecycle tracking and governance:
- **Experiment Tracking:** Logs hyperparameter dictionaries, metrics (MAE, MSE, RMSE, $R^2$), and artifacts for each candidate model run.
- **Artifact Repository:** Stores serialized model pipelines using cloudpickle.
- **Model Registry:** Provides a centralized model catalog (`PlayStore_RandomForestRegressor`, `PlayStore_LinearRegression`, etc.) tracking model lineage and versions for automated deployments.

---

### Q7: Why choose FastAPI over Flask or Django for model serving?
**Answer:**  
1. **Asynchronous Performance:** FastAPI is built on Starlette and Uvicorn (ASGI), providing asynchronous event loop handling capable of serving thousands of concurrent requests.
2. **Native Pydantic Data Validation:** Automatically validates incoming JSON data types (e.g. converting string numbers, enforcing required fields) and returns informative 422 HTTP errors for malformed payloads.
3. **Automatic OpenAPI / Swagger UI:** Automatically generates interactive documentation at `/docs` without external plugins.

---

### Q8: What is Population Stability Index (PSI) and how is it calculated?
**Answer:**  
PSI is a statistical metric used in risk management and MLOps to quantify how much a production feature distribution has shifted relative to a reference (training) baseline distribution.

**Formula:**
$$\text{PSI} = \sum_{i=1}^B (Q_i - P_i) \times \ln\left(\frac{Q_i + \epsilon}{P_i + \epsilon}\right)$$
Where:
- $B$ is the number of bins (typically 10 quantiles for continuous features).
- $P_i$ is the percentage of baseline samples in bin $i$.
- $Q_i$ is the percentage of production samples in bin $i$.
- $\epsilon$ is a small smoothing constant ($10^{-4}$) to prevent $\ln(0)$ or division by zero.

**Interpretation:**
- $\text{PSI} < 0.10$: **No Significant Drift** (distributions are stable).
- $0.10 \le \text{PSI} < 0.25$: **Moderate Drift** (slight shift; monitor closely).
- $\text{PSI} \ge 0.25$: **High Drift** (significant population change; trigger alert & retrain).

---

### Q9: Why did you use different statistical tests for numerical vs. categorical features?
**Answer:**  
Treating all features identically is statistically invalid:
- **Numerical Features (Reviews, Size, Installs, Price):** We utilize the **Two-Sample Kolmogorov-Smirnov (KS) test** (`scipy.stats.ks_2samp`). It compares the continuous Cumulative Distribution Functions (CDFs) of baseline and production samples to determine if they are drawn from the same underlying distribution ($p < 0.05$).
- **Categorical Features (Category, Type, Content Rating, Genres):** We utilize the **Chi-Square Goodness-of-Fit test** (`scipy.stats.chisquare`), which evaluates whether discrete category frequency counts significantly deviate from baseline proportions.

---

### Q10: How does your Docker containerization ensure production reproducibility?
**Answer:**  
1. **Lightweight Base Image:** We use `python:3.12-slim`, keeping the container footprint minimal ($\approx 250\text{MB}$).
2. **Dependency Isolation:** Dependencies are locked in `requirements-api.txt` and installed in an isolated container environment, eliminating "it works on my machine" issues.
3. **Layer Caching:** Docker builds copy dependencies first and source code second, enabling ultra-fast builds when application logic changes.
4. **Volume Mounts:** By mounting `-v /host/logs/inference:/app/logs/inference`, inference logs written by the container are saved to host storage, guaranteeing no log loss if a container restarts.
5. **Healthcheck:** The `Dockerfile` includes an automated `HEALTHCHECK` running `curl http://localhost:8000/health` every 30 seconds.

---

### Q11: How do you prevent duplicate email alert spam during persistent High Drift?
**Answer:**  
Our `EmailAlertService` implements a stateful **cooldown threshold** (default 30 minutes). When `HIGH DRIFT` is detected, the service checks the timestamp and list of affected features against the previous alert. If an alert was already dispatched within the cooldown window for identical features, subsequent alerts are suppressed and logged, protecting engineering inboxes from thousands of repetitive emails while keeping the monitoring engine running.

---

### Q12: What happens if an API user sends dirty or raw Play Store strings (e.g. Size "19M", Installs "10,000+", Price "$4.99")?
**Answer:**  
Our `api/preprocessing.py` module contains specialized input sanitation:
- `clean_size()`: Detects `"M"` (multiplies by 1.0) and `"k"` (divides by 1024), handling `"Varies with device"` with median imputation.
- `clean_installs()`: Strips commas and trailing `+` signs.
- `clean_price()`: Strips currency signs like `$`.
- `parse_date_features()`: Parses raw date strings (`"January 7, 2018"`) into numerical components (`Updated_Year`, `Updated_Month`, `Updated_Day`).

This allows the API to accept both real-world Play Store string dumps and pre-sanitized numeric payloads seamlessly.
