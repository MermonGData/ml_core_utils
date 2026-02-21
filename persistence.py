from pathlib import Path
import joblib
import json
import datetime

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
    

def save_artifact(content, name, folder_path, extension="pkl"):
    """Internal helper to save different file types."""
    path = folder_path / f"{name}.{extension}"
    if extension == "pkl":
        joblib.dump(content, path)
    elif extension == "csv":
        content.to_csv(path, index=False)
    elif extension == "json":
        with open(path, "w") as f:
            json.dump(content, f, indent=4)

def export_model_session(model, artifacts, metadata_info, base_path="../models", version="v1"):
    """
    Saves the model, associated data (X_train, etc.), and metadata.
    """
    export_path = Path(base_path) / version
    export_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save the Model
    save_artifact(model, "best_model", export_path, "pkl")
    
    # 2. Save Data/Artifacts (X_train, y_train, indices, etc.)
    for name, data in artifacts.items():
        save_artifact(data, name, export_path, "pkl")
        
    # 3. Save Metadata
    metadata = {
        "version": version,
        "date": datetime.datetime.now().isoformat(),
        "model_type": type(model).__name__,
        **metadata_info 
    }
    save_artifact(metadata, "metadata", export_path, "json")
    
    return export_path