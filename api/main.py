"""
FastAPI Prediction Service for Google Play Store App Rating Prediction
Provides real-time REST endpoints for rating predictions, health checks, and metadata.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from api.schemas import (
    AppInput,
    PredictionResponse,
    BatchAppInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from api.preprocessing import transform_input_to_features
from api.model_loader import model_service
from api.monitoring_logger import inference_logger

# Initialize FastAPI application
app = FastAPI(
    title="Google Play Store App Rating Prediction API",
    description="Production-ready MLOps API for predicting Google Play Store application ratings with real-time inference logging and drift monitoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web clients and dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Returns beginner-friendly, structured error messages for validation failures.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "ERROR",
            "message": "Input validation error: Please verify your JSON field formats.",
            "details": exc.errors()
        }
    )


@app.get("/", tags=["General"])
async def root():
    """
    Root status endpoint displaying API information and available navigation links.
    """
    return {
        "service": "Google Play Store App Rating Prediction API",
        "status": "online",
        "version": "1.0.0",
        "model_loaded": model_service.model is not None,
        "docs_url": "/docs",
        "web_app_ui": "/ui",
        "endpoints": {
            "web_ui": "GET /ui",
            "health_check": "GET /health",
            "model_info": "GET /model-info",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch"
        }
    }


@app.get("/ui", response_class=HTMLResponse, tags=["Web App"])
@app.get("/app", response_class=HTMLResponse, tags=["Web App"])
async def web_app_ui():
    """
    Interactive Web Application Graphical User Interface (GUI)
    Allows users to input application features through form fields and view predicted ratings.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Google Play Store Rating Predictor</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background: #0f172a; color: #f8fafc; min-height: 100vh; padding: 40px 20px; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .header p { color: #94a3b8; font-size: 0.95rem; margin-top: 8px; }
            .card { background: #1e293b; border-radius: 16px; border: 1px solid #334155; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }
            .form-group { display: flex; flex-direction: column; gap: 6px; }
            label { font-size: 0.85rem; font-weight: 600; color: #cbd5e1; }
            input, select { background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: #fff; padding: 10px 14px; font-size: 0.9rem; transition: border-color 0.2s; }
            input:focus, select:focus { outline: none; border-color: #38bdf8; ring: 2px #38bdf8; }
            .btn-predict { grid-column: 1 / -1; margin-top: 15px; background: linear-gradient(135deg, #0284c7, #6366f1); color: #fff; border: none; padding: 14px; font-size: 1.05rem; font-weight: 600; border-radius: 10px; cursor: pointer; transition: transform 0.1s, opacity 0.2s; }
            .btn-predict:hover { opacity: 0.95; transform: translateY(-1px); }
            .btn-predict:active { transform: translateY(0); }
            .result-card { margin-top: 25px; padding: 25px; border-radius: 14px; background: #0f172a; border: 1px solid #38bdf8; display: none; }
            .result-header { display: flex; justify-content: space-between; align-items: center; }
            .rating-badge { font-size: 2.8rem; font-weight: 700; color: #fbbf24; display: flex; align-items: center; gap: 8px; }
            .rating-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
            .meta-tags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
            .tag { font-size: 0.75rem; padding: 5px 10px; border-radius: 6px; background: #1e293b; border: 1px solid #475569; color: #94a3b8; }
            .links { text-align: center; margin-top: 20px; font-size: 0.85rem; color: #64748b; }
            .links a { color: #38bdf8; text-decoration: none; margin: 0 8px; }
            .links a:hover { text-decoration: underline; }
            .spinner { display: none; text-align: center; margin-top: 15px; color: #38bdf8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Google Play Store Rating Predictor</h1>
                <p>Production MLOps Regression Service with Real-Time Inference & Drift Logging</p>
            </div>

            <div class="card">
                <form id="predictionForm" class="grid">
                    <div class="form-group" style="grid-column: 1 / -1;">
                        <label for="app_name">Application Name</label>
                        <input type="text" id="app_name" value="Subway Surfers" required>
                    </div>

                    <div class="form-group">
                        <label for="category">Category</label>
                        <select id="category">
                            <option value="GAME" selected>GAME</option>
                            <option value="FAMILY">FAMILY</option>
                            <option value="TOOLS">TOOLS</option>
                            <option value="PRODUCTIVITY">PRODUCTIVITY</option>
                            <option value="PHOTOGRAPHY">PHOTOGRAPHY</option>
                            <option value="EDUCATION">EDUCATION</option>
                            <option value="FINANCE">FINANCE</option>
                            <option value="SOCIAL">SOCIAL</option>
                            <option value="COMMUNICATION">COMMUNICATION</option>
                            <option value="HEALTH_AND_FITNESS">HEALTH_AND_FITNESS</option>
                            <option value="ART_AND_DESIGN">ART_AND_DESIGN</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="reviews">Number of Reviews</label>
                        <input type="number" id="reviews" value="27722264" required>
                    </div>

                    <div class="form-group">
                        <label for="size">App Size (e.g. 76M, 500k)</label>
                        <input type="text" id="size" value="76M" required>
                    </div>

                    <div class="form-group">
                        <label for="installs">Estimated Installs</label>
                        <select id="installs">
                            <option value="1,000,000,000+" selected>1,000,000,000+</option>
                            <option value="500,000,000+">500,000,000+</option>
                            <option value="100,000,000+">100,000,000+</option>
                            <option value="10,000,000+">10,000,000+</option>
                            <option value="1,000,000+">1,000,000+</option>
                            <option value="100,000+">100,000+</option>
                            <option value="10,000+">10,000+</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="app_type">Monetization Type</label>
                        <select id="app_type">
                            <option value="Free" selected>Free</option>
                            <option value="Paid">Paid</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="price">Price (USD)</label>
                        <input type="text" id="price" value="0" required>
                    </div>

                    <div class="form-group">
                        <label for="content_rating">Content Rating</label>
                        <select id="content_rating">
                            <option value="Everyone" selected>Everyone</option>
                            <option value="Everyone 10+">Everyone 10+</option>
                            <option value="Teen">Teen</option>
                            <option value="Mature 17+">Mature 17+</option>
                            <option value="Adults only 18+">Adults only 18+</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="genres">Primary Genre</label>
                        <input type="text" id="genres" value="Arcade" required>
                    </div>

                    <div class="form-group">
                        <label for="android_ver">Minimum Android Version</label>
                        <input type="text" id="android_ver" value="4.1 and up" required>
                    </div>

                    <button type="submit" class="btn-predict">Predict Application Rating</button>
                </form>

                <div class="spinner" id="spinner">Calculating prediction using Random Forest Regressor...</div>

                <div class="result-card" id="resultCard">
                    <div class="result-header">
                        <div>
                            <div class="rating-label">Predicted Rating (1.0 to 5.0)</div>
                            <div class="rating-badge" id="predictedRating">-- <span style="font-size: 1.8rem;">★</span></div>
                        </div>
                        <div style="text-align: right;">
                            <span class="tag" style="background: #065f46; color: #34d399; font-weight: 600;" id="statusBadge">SUCCESS</span>
                        </div>
                    </div>
                    <div class="meta-tags">
                        <span class="tag" id="latencyTag">Latency: -- ms</span>
                        <span class="tag" id="modelTag">Model: Google_Play_Store_RandomForest</span>
                        <span class="tag" id="requestTag">Request ID: --</span>
                    </div>
                </div>
            </div>

            <div class="links">
                <a href="/docs" target="_blank">Swagger Documentation (/docs)</a> &bull;
                <a href="/health" target="_blank">Health Check (/health)</a> &bull;
                <a href="/model-info" target="_blank">Model Info (/model-info)</a>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const spinner = document.getElementById('spinner');
                const resultCard = document.getElementById('resultCard');

                spinner.style.display = 'block';
                resultCard.style.display = 'none';

                const payload = {
                    "App": document.getElementById('app_name').value,
                    "Category": document.getElementById('category').value,
                    "Reviews": document.getElementById('reviews').value,
                    "Size": document.getElementById('size').value,
                    "Installs": document.getElementById('installs').value,
                    "Type": document.getElementById('app_type').value,
                    "Price": document.getElementById('price').value,
                    "Content Rating": document.getElementById('content_rating').value,
                    "Genres": document.getElementById('genres').value,
                    "Android Ver": document.getElementById('android_ver').value
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();
                    spinner.style.display = 'none';

                    if (response.ok) {
                        document.getElementById('predictedRating').innerHTML = data.predicted_rating + ' <span style="font-size: 1.8rem; color: #fbbf24;">★</span>';
                        document.getElementById('latencyTag').textContent = 'Latency: ' + data.prediction_latency_ms + ' ms';
                        document.getElementById('modelTag').textContent = 'Model: ' + data.model_name + ' (v' + data.model_version + ')';
                        document.getElementById('requestTag').textContent = 'Request ID: ' + data.request_id.substring(0, 8) + '...';
                        resultCard.style.display = 'block';
                    } else {
                        alert('Prediction Error: ' + (data.message || JSON.stringify(data.details)));
                    }
                } catch (err) {
                    spinner.style.display = 'none';
                    alert('Network error connecting to API: ' + err.message);
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Liveness and readiness health check for container orchestration and uptime monitoring.
    """
    return HealthResponse(
        status="healthy" if model_service.model is not None else "degraded",
        model_loaded=model_service.model is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Returns registered model metadata, pipeline architecture, and evaluation metrics.
    """
    try:
        info = model_service.get_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_rating(app_input: AppInput, request: Request):
    """
    Predicts the Google Play Store rating for a single mobile application.
    Logs inference features, latency, and predictions for data drift monitoring.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    input_dict = app_input.model_dump(by_alias=True)

    try:
        # Preprocess input into feature DataFrame
        df_features = transform_input_to_features(input_dict)

        # Generate model prediction
        predicted_rating = model_service.predict_dataframe(df_features)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Log prediction to JSONL inference log for drift monitoring
        inference_logger.log_prediction(
            request_id=request_id,
            input_features=input_dict,
            predicted_rating=predicted_rating,
            prediction_latency_ms=latency_ms,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            status="SUCCESS",
            client_ip=client_ip
        )

        return PredictionResponse(
            request_id=request_id,
            app_name=str(input_dict.get("App", "Unknown App")),
            predicted_rating=round(predicted_rating, 2),
            prediction_latency_ms=round(latency_ms, 2),
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            status="SUCCESS"
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Log failure event
        inference_logger.log_prediction(
            request_id=request_id,
            input_features=input_dict,
            predicted_rating=-1.0,
            prediction_latency_ms=latency_ms,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            status=f"FAILED: {str(e)}",
            client_ip=client_ip
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch_input: BatchAppInput, request: Request):
    """
    Generates rating predictions for multiple applications in a single batch request.
    """
    start_time = time.perf_counter()
    batch_request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    predictions = []

    for item in batch_input.applications:
        item_start = time.perf_counter()
        req_id = str(uuid.uuid4())
        item_dict = item.model_dump(by_alias=True)

        try:
            df_feat = transform_input_to_features(item_dict)
            rating = model_service.predict_dataframe(df_feat)
            item_latency = (time.perf_counter() - item_start) * 1000.0

            inference_logger.log_prediction(
                request_id=req_id,
                input_features=item_dict,
                predicted_rating=rating,
                prediction_latency_ms=item_latency,
                model_name=model_service.model_name,
                model_version=model_service.model_version,
                status="SUCCESS",
                client_ip=client_ip
            )

            predictions.append(
                PredictionResponse(
                    request_id=req_id,
                    app_name=str(item_dict.get("App", "Unknown App")),
                    predicted_rating=round(rating, 2),
                    prediction_latency_ms=round(item_latency, 2),
                    model_name=model_service.model_name,
                    model_version=model_service.model_version,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    status="SUCCESS"
                )
            )
        except Exception as e:
            item_latency = (time.perf_counter() - item_start) * 1000.0
            predictions.append(
                PredictionResponse(
                    request_id=req_id,
                    app_name=str(item_dict.get("App", "Unknown App")),
                    predicted_rating=0.0,
                    prediction_latency_ms=round(item_latency, 2),
                    model_name=model_service.model_name,
                    model_version=model_service.model_version,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    status=f"FAILED: {str(e)}"
                )
            )

    total_latency = (time.perf_counter() - start_time) * 1000.0

    return BatchPredictionResponse(
        request_id=batch_request_id,
        total_applications=len(predictions),
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        status="SUCCESS"
    )
