
# ============================================================
# GOOGLE PLAY STORE RATING PREDICTION
# PREPROCESSING
# ============================================================

import os
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. CLEAN SIZE
# ============================================================

def clean_size(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() == "varies with device":
        return np.nan

    try:

        if value.lower().endswith("m"):
            return float(value[:-1])

        elif value.lower().endswith("k"):
            return float(value[:-1]) / 1024

        else:
            return float(value)

    except (ValueError, TypeError):
        return np.nan


# ============================================================
# 2. CLEAN REVIEWS
# ============================================================

def clean_reviews(value):

    if pd.isna(value):
        return np.nan

    try:

        return float(
            str(value)
            .replace(",", "")
            .strip()
        )

    except (ValueError, TypeError):
        return np.nan


# ============================================================
# 3. CLEAN INSTALLS
# ============================================================

def clean_installs(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:

        value = value.replace(",", "")
        value = value.replace("+", "")

        return float(value)

    except (ValueError, TypeError):

        # Handles invalid values such as "Free"
        return np.nan


# ============================================================
# 4. CLEAN PRICE
# ============================================================

def clean_price(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:

        value = value.replace("$", "")

        return float(value)

    except (ValueError, TypeError):
        return np.nan


# ============================================================
# 5. CLEAN CURRENT VERSION
# ============================================================

def clean_version(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() == "varies with device":
        return np.nan

    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if match:

        try:
            return float(match.group())

        except ValueError:
            return np.nan

    return np.nan


# ============================================================
# 6. CLEAN ANDROID VERSION
# ============================================================

def clean_android_version(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() == "varies with device":
        return np.nan

    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if match:

        try:
            return float(match.group())

        except ValueError:
            return np.nan

    return np.nan


# ============================================================
# 7. PREPROCESS DATA
# ============================================================

def preprocess_data(
    input_path="data/raw/googleplaystore.csv",
    output_path="data/processed/processed_googleplaystore.csv",
    metadata_path="data/processed/dataset_metadata.json"
):

    print("=" * 60)
    print("GOOGLE PLAY STORE RATING ML PIPELINE")
    print("=" * 60)

    print("\nSTEP 1: PREPROCESSING")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"\nDataset not found:\n{input_path}\n\n"
            "Please place googleplaystore.csv inside:\n"
            "data/raw/"
        )

    df = pd.read_csv(input_path)

    print("Dataset loaded successfully!")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # ========================================================
    # DISPLAY COLUMNS
    # ========================================================

    print("\nDataset columns:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(i, ".", column)

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    duplicate_count = df.duplicated().sum()

    print(
        "\nDuplicate rows:",
        duplicate_count
    )

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Shape after removing duplicates:",
        df.shape
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [

        "App",
        "Category",
        "Rating",
        "Reviews",
        "Size",
        "Installs",
        "Type",
        "Price",
        "Content Rating",
        "Genres",
        "Last Updated",
        "Current Ver",
        "Android Ver"

    ]

    missing_columns = [

        column
        for column in required_columns
        if column not in df.columns

    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + str(missing_columns)
        )

    # ========================================================
    # CLEAN REVIEWS
    # ========================================================

    print("\nCleaning Reviews...")

    df["Reviews"] = (
        df["Reviews"]
        .apply(clean_reviews)
    )

    # ========================================================
    # CLEAN SIZE
    # ========================================================

    print("Cleaning Size...")

    df["Size"] = (
        df["Size"]
        .apply(clean_size)
    )

    # ========================================================
    # CLEAN INSTALLS
    # ========================================================

    print("Cleaning Installs...")

    df["Installs"] = (
        df["Installs"]
        .apply(clean_installs)
    )

    # ========================================================
    # CLEAN PRICE
    # ========================================================

    print("Cleaning Price...")

    df["Price"] = (
        df["Price"]
        .apply(clean_price)
    )

    # ========================================================
    # CLEAN CURRENT VERSION
    # ========================================================

    print("Cleaning Current Ver...")

    df["Current Ver"] = (
        df["Current Ver"]
        .apply(clean_version)
    )

    # ========================================================
    # CLEAN ANDROID VERSION
    # ========================================================

    print("Cleaning Android Ver...")

    df["Android Ver"] = (
        df["Android Ver"]
        .apply(clean_android_version)
    )

    # ========================================================
    # CLEAN RATING
    # ========================================================

    print("Cleaning Rating...")

    df["Rating"] = pd.to_numeric(
        df["Rating"],
        errors="coerce"
    )

    # ========================================================
    # REMOVE INVALID RATINGS
    # ========================================================

    invalid_ratings = (

        (df["Rating"] < 1) |
        (df["Rating"] > 5)

    )

    print(
        "Invalid ratings:",
        invalid_ratings.sum()
    )

    df.loc[
        invalid_ratings,
        "Rating"
    ] = np.nan

    # ========================================================
    # REMOVE ROWS WITHOUT TARGET
    # ========================================================

    missing_rating = (
        df["Rating"].isna().sum()
    )

    print(
        "Missing Rating values:",
        missing_rating
    )

    df = (
        df
        .dropna(
            subset=["Rating"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # PROCESS LAST UPDATED
    # ========================================================

    print(
        "Processing Last Updated..."
    )

    df["Last Updated"] = pd.to_datetime(
        df["Last Updated"],
        errors="coerce"
    )

    df["Updated_Year"] = (
        df["Last Updated"].dt.year
    )

    df["Updated_Month"] = (
        df["Last Updated"].dt.month
    )

    df["Updated_Day"] = (
        df["Last Updated"].dt.day
    )

    # Remove original date column

    df = df.drop(
        columns=["Last Updated"]
    )

    # ========================================================
    # CLEAN CATEGORICAL COLUMNS
    # ========================================================

    categorical_columns = [

        "App",
        "Category",
        "Type",
        "Content Rating",
        "Genres"

    ]

    print(
        "Cleaning categorical columns..."
    )

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = (
            df[column]
            .fillna("Unknown")
        )

    # ========================================================
    # FILL NUMERICAL MISSING VALUES
    # ========================================================

    numerical_columns = [

        "Reviews",
        "Size",
        "Installs",
        "Price",
        "Current Ver",
        "Android Ver",
        "Updated_Year",
        "Updated_Month",
        "Updated_Day"

    ]

    print(
        "Filling numerical missing values..."
    )

    for column in numerical_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            median_value = (
                df[column].median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )

    # ========================================================
    # FINAL DATA CHECK
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL DATA CHECK")
    print("=" * 60)

    print(
        "Final rows:",
        df.shape[0]
    )

    print(
        "Final columns:",
        df.shape[1]
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    # ========================================================
    # CREATE PROCESSED DIRECTORY
    # ========================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # ========================================================
    # SAVE PROCESSED DATASET
    # ========================================================

    print(
        "\nSaving processed dataset..."
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "Processed dataset saved:"
    )

    print(
        output_path
    )

    # ========================================================
    # CREATE METADATA
    # ========================================================

    metadata = {

        "dataset_name":
            "Google Play Store",

        "problem_type":
            "Supervised Machine Learning",

        "task":
            "Regression",

        "target_variable":
            "Rating",

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "features":
            [
                column
                for column in df.columns
                if column != "Rating"
            ],

        "rating_minimum":
            float(
                df["Rating"].min()
            ),

        "rating_maximum":
            float(
                df["Rating"].max()
            )

    }

    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "Metadata saved:"
    )

    print(
        metadata_path
    )

    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_data()

