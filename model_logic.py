import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def prepare_feature_frame(data):
    df = data.copy()
    df["Position"] = pd.to_numeric(df.get("Position"), errors="coerce")
    df["GridPosition"] = pd.to_numeric(df.get("GridPosition"), errors="coerce")
    df["practice_best_lap_time"] = pd.to_numeric(df.get("practice_best_lap_time"), errors="coerce")
    df["practice_mean_lap_time"] = pd.to_numeric(df.get("practice_mean_lap_time"), errors="coerce")
    df["practice_lap_count"] = pd.to_numeric(df.get("practice_lap_count"), errors="coerce")
    df["qualifying_best_lap_time"] = pd.to_numeric(df.get("qualifying_best_lap_time"), errors="coerce")
    df["qualifying_mean_lap_time"] = pd.to_numeric(df.get("qualifying_mean_lap_time"), errors="coerce")
    df["qualifying_lap_count"] = pd.to_numeric(df.get("qualifying_lap_count"), errors="coerce")
    df["air_temp"] = pd.to_numeric(df.get("air_temp"), errors="coerce")
    df["track_temp"] = pd.to_numeric(df.get("track_temp"), errors="coerce")
    df["humidity"] = pd.to_numeric(df.get("humidity"), errors="coerce")
    df["rainfall"] = pd.to_numeric(df.get("rainfall"), errors="coerce")
    df["TeamName"] = df.get("TeamName", "Unknown").fillna("Unknown").astype(str)
    df["FullName"] = df.get("FullName", "Unknown").fillna("Unknown").astype(str)
    df["podium"] = (df["Position"] <= 3).astype(int)

    feature_frame = pd.DataFrame({
        "GridPosition": df["GridPosition"],
        "practice_best_lap_time": df["practice_best_lap_time"],
        "practice_mean_lap_time": df["practice_mean_lap_time"],
        "practice_lap_count": df["practice_lap_count"],
        "qualifying_best_lap_time": df["qualifying_best_lap_time"],
        "qualifying_mean_lap_time": df["qualifying_mean_lap_time"],
        "qualifying_lap_count": df["qualifying_lap_count"],
        "air_temp": df["air_temp"],
        "track_temp": df["track_temp"],
        "humidity": df["humidity"],
        "rainfall": df["rainfall"],
        "TeamName": df["TeamName"],
        "FullName": df["FullName"],
    })

    feature_frame = pd.get_dummies(feature_frame, columns=["TeamName", "FullName"], dummy_na=False)
    feature_frame = feature_frame.fillna(0)
    return feature_frame


def train_british_gp_model(data):
    df = data.copy()
    if df.empty:
        raise ValueError("No training data available")

    feature_frame = prepare_feature_frame(df)
    valid = df["Position"].notna() & df["GridPosition"].notna()
    X = feature_frame.loc[valid].reset_index(drop=True)
    y_podium = df.loc[valid, "Position"].le(3).astype(int).reset_index(drop=True)
    y_pos = df.loc[valid, "Position"].astype(float).reset_index(drop=True)

    X_train, X_test, y_pod_tr, y_pod_te, y_pos_tr, y_pos_te = train_test_split(
        X, y_podium, y_pos, test_size=0.2, random_state=42
    )

    clf = xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, eval_metric="logloss")
    clf.fit(X_train, y_pod_tr)

    reg = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, max_depth=4)
    reg.fit(X_train, y_pos_tr)

    pod_pred = clf.predict(X_test)
    pos_pred = reg.predict(X_test)

    metrics = {
        "podium_accuracy": accuracy_score(y_pod_te, pod_pred),
        "precision": precision_score(y_pod_te, pod_pred, zero_division=0),
        "recall": recall_score(y_pod_te, pod_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_pod_te, clf.predict_proba(X_test)[:, 1]),
        "mae": mean_absolute_error(y_pos_te, pos_pred),
        "position_rmse": float(np.sqrt(mean_squared_error(y_pos_te, pos_pred))),
        "model_clf": clf,
        "model_reg": reg,
        "feature_columns": X.columns.tolist(),
    }
    return metrics


def predict_record(model_bundle, record):
    feature_frame = prepare_feature_frame(pd.DataFrame([record]))
    feature_frame = feature_frame.reindex(columns=model_bundle["feature_columns"], fill_value=0)
    podium_probability = float(model_bundle["model_clf"].predict_proba(feature_frame)[0, 1])
    predicted_position = float(model_bundle["model_reg"].predict(feature_frame)[0])
    return podium_probability, predicted_position