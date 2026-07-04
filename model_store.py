import os
import json
import joblib
from datetime import datetime
import pandas as pd
import model_logic

MODELS_DIR = "models"
CLF_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
REG_PATH = os.path.join(MODELS_DIR, "regressor.pkl")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

def get_or_train_models(data, force_retrain=False):
    """
    Retrieves saved models if available and valid.
    Otherwise, trains new models and saves them.
    """
    if force_retrain or not models_exist():
        print("Training new models...")
        bundle = model_logic.train_british_gp_model(data)
        save_models(bundle)
        return bundle
        
    # Check if feature schema changed
    current_features = model_logic.prepare_feature_frame(data).columns.tolist()
    if schema_changed(current_features):
        print("Feature schema changed, retraining models...")
        bundle = model_logic.train_british_gp_model(data)
        save_models(bundle)
        return bundle
        
    print("Loading saved models...")
    bundle = load_models()
    if bundle is None:
        print("Failed to load models, retraining...")
        bundle = model_logic.train_british_gp_model(data)
        save_models(bundle)
        return bundle
        
    return bundle

def models_exist():
    """Check if all necessary model files exist."""
    return os.path.exists(CLF_PATH) and os.path.exists(REG_PATH) and os.path.exists(META_PATH)
    
def schema_changed(current_features):
    """Check if the feature schema has changed compared to saved metadata."""
    if not os.path.exists(META_PATH):
        return True
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        return meta.get("feature_columns") != current_features
    except Exception:
        return True

def load_models():
    """Load models and metadata from disk."""
    try:
        clf = joblib.load(CLF_PATH)
        reg = joblib.load(REG_PATH)
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            
        return {
            "model_clf": clf,
            "model_reg": reg,
            "feature_columns": meta["feature_columns"],
            "podium_accuracy": meta["metrics"]["podium_accuracy"],
            "precision": meta["metrics"]["precision"],
            "recall": meta["metrics"]["recall"],
            "roc_auc": meta["metrics"]["roc_auc"],
            "mae": meta["metrics"]["mae"],
            "position_rmse": meta["metrics"]["position_rmse"],
            "feature_importance": pd.Series(meta.get("feature_importance", {}))
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
        
def save_models(model_bundle):
    """Save models and metadata to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the sklearn/xgboost models using joblib
    joblib.dump(model_bundle["model_clf"], CLF_PATH)
    joblib.dump(model_bundle["model_reg"], REG_PATH)
    
    # Save metadata including metrics and feature schema
    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0",
        "feature_columns": model_bundle["feature_columns"],
        "metrics": {
            "podium_accuracy": model_bundle["podium_accuracy"],
            "precision": model_bundle["precision"],
            "recall": model_bundle["recall"],
            "roc_auc": model_bundle["roc_auc"],
            "mae": model_bundle["mae"],
            "position_rmse": model_bundle["position_rmse"],
        },
        "feature_importance": model_bundle["feature_importance"].to_dict() if model_bundle["feature_importance"] is not None else {}
    }
    
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
