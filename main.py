import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import data_loader
import model_logic

app = FastAPI(title="British GP Predictor", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("templates/index.html")


@app.get("/api/predictions")
def get_predictions():
    training_data = data_loader.get_british_gp_data(years=(2021, 2022, 2023, 2024, 2025))
    bundle = model_logic.train_british_gp_model(training_data)
    prediction_frame = data_loader.get_upcoming_british_gp_prediction(historical_data=training_data)

    if prediction_frame.empty:
        return {"drivers": []}

    scored_rows = []
    for _, row in prediction_frame.iterrows():
        podium_probability, predicted_position = model_logic.predict_record(bundle, row.to_dict())
        scored_rows.append({
            "fullName": row.get("FullName", ""),
            "teamName": row.get("TeamName", ""),
            "gridPosition": row.get("GridPosition", None),
            "podiumProbability": float(podium_probability),
            "predictedFinish": int(round(predicted_position)),
        })

    scored_df = pd.DataFrame(scored_rows).sort_values("podiumProbability", ascending=False)
    return {"drivers": scored_df.to_dict(orient="records"), "metrics": {
        "accuracy": float(bundle["podium_accuracy"]),
        "precision": float(bundle["precision"]),
        "recall": float(bundle["recall"]),
        "rocAuc": float(bundle["roc_auc"]),
        "mae": float(bundle["mae"]),
    }}
