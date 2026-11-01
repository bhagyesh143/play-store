"""
Pydantic Schemas for Google Play Store Rating Prediction API
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class AppInput(BaseModel):
    """
    Input schema representing application features submitted for rating prediction.
    Accepts both raw Google Play Store string formats and numeric equivalents.
    """
    App: str = Field(
        default="Photo Editor & Candy Camera",
        description="Name of the application"
    )
    Category: str = Field(
        default="ART_AND_DESIGN",
        description="App category (e.g. ART_AND_DESIGN, FAMILY, GAME, TOOLS)"
    )
    Reviews: Union[int, float, str] = Field(
        default=159,
        description="Number of user reviews (e.g. 159 or '159')"
    )
    Size: Union[float, int, str] = Field(
        default="19M",
        description="App size in Megabytes, or raw string format (e.g. '19M', '500k', 19.0)"
    )
    Installs: Union[float, int, str] = Field(
        default="10,000+",
        description="Total number of installs (e.g. '10,000+' or 10000)"
    )
    Type: str = Field(
        default="Free",
        description="App monetization type ('Free' or 'Paid')"
    )
    Price: Union[float, int, str] = Field(
        default="0",
        description="Price in USD (e.g. '$4.99', '0', or 0.0)"
    )
    Content_Rating: str = Field(
        default="Everyone",
        alias="Content Rating",
        description="Target audience content rating (e.g. 'Everyone', 'Teen', 'Mature 17+')"
    )
    Genres: str = Field(
        default="Art & Design",
        description="Primary and secondary genres (e.g. 'Art & Design', 'Action', 'Puzzle')"
    )
    Last_Updated: Optional[str] = Field(
        default="January 7, 2018",
        alias="Last Updated",
        description="Date of the last update (e.g. 'January 7, 2018' or '2018-01-07')"
    )
    Current_Ver: Optional[Union[float, str]] = Field(
        default="1.0.0",
        alias="Current Ver",
        description="Current application release version (e.g. '1.0.0')"
    )
    Android_Ver: Optional[Union[float, str]] = Field(
        default="4.0.3 and up",
        alias="Android Ver",
        description="Minimum supported Android OS version (e.g. '4.0.3 and up')"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "App": "Photo Editor & Candy Camera",
                "Category": "ART_AND_DESIGN",
                "Reviews": 159,
                "Size": "19M",
                "Installs": "10,000+",
                "Type": "Free",
                "Price": "0",
                "Content Rating": "Everyone",
                "Genres": "Art & Design",
                "Last Updated": "January 7, 2018",
                "Current Ver": "1.0.0",
                "Android Ver": "4.0.3 and up"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema returning the predicted rating and inference metadata.
    """
    request_id: str = Field(..., description="Unique UUID for this inference request")
    app_name: str = Field(..., description="Name of the application evaluated")
    predicted_rating: float = Field(..., description="Predicted Google Play Store rating (1.0 to 5.0)")
    prediction_latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_name: str = Field(..., description="Name of the active model")
    model_version: str = Field(..., description="Registered model version")
    timestamp: str = Field(..., description="ISO 8601 inference timestamp")
    status: str = Field(default="SUCCESS", description="Inference status")


class BatchAppInput(BaseModel):
    """
    Batch input schema containing a list of applications.
    """
    applications: List[AppInput] = Field(..., description="List of applications to predict")


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response schema.
    """
    request_id: str
    total_applications: int
    predictions: List[PredictionResponse]
    total_latency_ms: float
    status: str = "SUCCESS"


class HealthResponse(BaseModel):
    """
    API health check response schema.
    """
    status: str = Field(default="healthy", description="Service health state")
    model_loaded: bool = Field(..., description="Whether model is in memory and ready")
    version: str = Field(..., description="API service version")
    timestamp: str = Field(..., description="Current system timestamp")


class ModelInfoResponse(BaseModel):
    """
    Model details, performance metrics, and feature schemas.
    """
    model_name: str
    model_version: str
    problem_type: str
    target_variable: str
    rating_range: Dict[str, float]
    baseline_metrics: Dict[str, Any]
    features_expected: List[str]
    pipeline_steps: List[str]
