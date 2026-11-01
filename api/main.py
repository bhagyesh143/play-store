"""
FastAPI Prediction Service for Google Play Store App Rating Prediction
Provides real-time REST endpoints for rating predictions, health checks, and metadata.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
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
        "endpoints": {
            "health_check": "GET /health",
            "model_info": "GET /model-info",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch"
        }
    }


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
