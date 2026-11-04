import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

def get_validation_schema():
    return pa.DataFrameSchema({
        "App": Column(pa.String),
        "Category": Column(pa.String),
        "Reviews": Column(pa.String),
        "Size": Column(pa.String),
        "Installs": Column(pa.String),
        "Type": Column(pa.String, nullable=True),
        "Price": Column(pa.String),
        "Content Rating": Column(pa.String, nullable=True),
        "Genres": Column(pa.String),
    }, strict=False)
