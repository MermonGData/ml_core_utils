from pathlib import Path
import joblib
import json
from datetime import datetime

# --- UTILITIES ---

def _save_file(content, name, folder_path, extension="pkl"):
    """Generic internal helper to route different file types."""
    path = Path(folder_path) / f"{name}.{extension}"
    
    if extension == "pkl":
        joblib.dump(content, path)
    elif extension == "csv":
        content.to_csv(path, index=False)
    elif extension == "json":
        with open(path, "w") as f:
            json.dump(content, f, indent=4)
    return path

# --- LOADING FUNCTIONS ---

def load_model_artifact(file_path):
    """Loads pickled ML artifacts (models, scalers, arrays)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No artifact found at {path}")
    return joblib.load(path)

def load_multiple_artifacts(directory, filenames):
    """Helper to load a list of files from the same folder into a dict."""
    dir_path = Path(directory)
    return {f: joblib.load(dir_path / f) for f in filenames}

# --- EXPORT FUNCTIONS (The Two Versions) ---

def export_model_session(model, artifacts, metadata_info, base_path="../models", version="v1"):
    """
    Saves model, arbitrary dict of artifacts as pickles, and metadata.
    """
    export_path = Path(base_path) / version
    export_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save the Model
    _save_file(model, "best_model", export_path, "pkl")
    
    # 2. Save Data/Artifacts (X_train, y_train, etc.)
    for name, data in artifacts.items():
        _save_file(data, name, export_path, "pkl")
        
    # 3. Save Metadata
    metadata = {
        "version": version,
        "date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        **metadata_info 
    }
    _save_file(metadata, "metadata", export_path, "json")
    
    return export_path

def save_model_artifact(model, metrics, features, base_path="../models", version="v1"):
    """
    Saves model, metrics as CSVs, and explicit feature list.
    """
    export_path = Path(base_path) / version
    export_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model
    _save_file(model, "model", export_path, "pkl")
    
    # 2. Save metrics (as CSVs)
    for name, df in metrics.items():
        _save_file(df, name, export_path, "csv")
    
    # 3. Save metadata
    metadata = {
        "version": version,
        "date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "features": list(features)
    }
    _save_file(metadata, "metadata", export_path, "json")
    
    return export_path