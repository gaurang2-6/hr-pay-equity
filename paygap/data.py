"""Download the IBM HR Analytics Employee Attrition dataset (cached in data/)."""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# IBM's own copy of the Kaggle "IBM HR Analytics Employee Attrition & Performance" file.
SOURCE = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
FILE = "ibm_hr_attrition.csv"


def fetch() -> Path:
    path = DATA_DIR / FILE
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {FILE}")
        urlretrieve(SOURCE, path)
    return path


def employees() -> pd.DataFrame:
    """One row per employee, plus the columns the models use (log pay, a female dummy)."""
    df = pd.read_csv(fetch(), encoding="utf-8-sig")
    df["log_pay"] = np.log(df.MonthlyIncome)
    df["female"] = (df.Gender == "Female").astype(int)
    return df
