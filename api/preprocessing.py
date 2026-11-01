"""
Robust Preprocessing and Feature Engineering Utility for FastAPI
Transforms raw API inputs into the schema required by the trained model pipeline.
"""

import re
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any


def clean_size(value: Any) -> float:
    """
    Converts app size strings (e.g. '19M', '500k', 'Varies with device') into Megabytes (float).
    """
    if pd.isna(value) or value is None:
        return 15.0  # Median default

    val_str = str(value).strip().lower()

    if "varies" in val_str or val_str == "nan":
        return 15.0

    try:
        if val_str.endswith("m"):
            return float(val_str[:-1])
        elif val_str.endswith("k"):
            return float(val_str[:-1]) / 1024.0
        else:
            return float(val_str)
    except (ValueError, TypeError):
        return 15.0


def clean_reviews(value: Any) -> float:
    """
    Cleans review count to float.
    """
    if pd.isna(value) or value is None:
        return 100.0

    try:
        val_str = str(value).replace(",", "").strip()
        return float(val_str)
    except (ValueError, TypeError):
        return 100.0


def clean_installs(value: Any) -> float:
    """
    Converts install strings like '10,000+' or '5,000,000+' into float count.
    """
    if pd.isna(value) or value is None:
        return 10000.0

    try:
        val_str = str(value).replace(",", "").replace("+", "").strip()
        return float(val_str)
    except (ValueError, TypeError):
        return 10000.0


def clean_price(value: Any) -> float:
    """
    Strips currency symbols like '$' and converts to float price in USD.
    """
    if pd.isna(value) or value is None:
        return 0.0

    try:
        val_str = str(value).replace("$", "").strip()
        return float(val_str)
    except (ValueError, TypeError):
        return 0.0


def clean_version(value: Any) -> float:
    """
    Extracts leading major.minor version digits (e.g. '1.0.0' -> 1.0).
    """
    if pd.isna(value) or value is None:
        return 2.0

    val_str = str(value).strip().lower()
    if "varies" in val_str:
        return 2.0

    match = re.search(r"\d+(?:\.\d+)?", val_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 2.0
    return 2.0


def clean_android_version(value: Any) -> float:
    """
    Extracts Android version floor (e.g. '4.0.3 and up' -> 4.0).
    """
    if pd.isna(value) or value is None:
        return 4.0

    val_str = str(value).strip().lower()
    if "varies" in val_str:
        return 4.0

    match = re.search(r"\d+(?:\.\d+)?", val_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 4.0
    return 4.0


def parse_date_features(date_str: Any) -> tuple:
    """
    Extracts (Year, Month, Day) from 'Last Updated' strings.
    """
    default_now = datetime.datetime.now()
    default_tuple = (default_now.year, default_now.month, default_now.day)

    if pd.isna(date_str) or not date_str:
        return default_tuple

    try:
        parsed = pd.to_datetime(str(date_str).strip(), errors="coerce")
        if pd.isna(parsed):
            return default_tuple
        return (int(parsed.year), int(parsed.month), int(parsed.day))
    except Exception:
        return default_tuple


def transform_input_to_features(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms a single or batch raw dictionary into a DataFrame ready for the model pipeline.
    """
    # Normalize keys if aliases were passed
    app_name = str(raw_input.get("App", "Unknown App")).strip()
    category = str(raw_input.get("Category", "FAMILY")).strip()
    app_type = str(raw_input.get("Type", "Free")).strip()
    content_rating = str(raw_input.get("Content Rating", raw_input.get("Content_Rating", "Everyone"))).strip()
    genres = str(raw_input.get("Genres", "Tools")).strip()

    # Numerical cleans
    reviews = clean_reviews(raw_input.get("Reviews", 100))
    size = clean_size(raw_input.get("Size", 15.0))
    installs = clean_installs(raw_input.get("Installs", 10000))
    price = clean_price(raw_input.get("Price", 0.0))

    # Version cleans
    current_ver = clean_version(raw_input.get("Current Ver", raw_input.get("Current_Ver", "1.0")))
    android_ver = clean_android_version(raw_input.get("Android Ver", raw_input.get("Android_Ver", "4.0")))

    # Date parse
    last_updated = raw_input.get("Last Updated", raw_input.get("Last_Updated", None))
    year, month, day = parse_date_features(last_updated)

    feature_dict = {
        "App": [app_name],
        "Category": [category],
        "Reviews": [reviews],
        "Size": [size],
        "Installs": [installs],
        "Type": [app_type],
        "Price": [price],
        "Content Rating": [content_rating],
        "Genres": [genres],
        "Current Ver": [current_ver],
        "Android Ver": [android_ver],
        "Updated_Year": [year],
        "Updated_Month": [month],
        "Updated_Day": [day]
    }

    return pd.DataFrame(feature_dict)
