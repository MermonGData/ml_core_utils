from pathlib import Path
import pyreadstat
import pandas as pd


def load_data(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        return df, None
    
    elif ext == ".sav":
        df, meta = pyreadstat.read_sav(path)
        return df, meta
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    

