import argparse
import pandas as pd
import data_loader, model_logic
import model_store

parser = argparse.ArgumentParser(description="Validate British GP models.")
parser.add_argument("--retrain", action="store_true", help="Force retrain models")
args = parser.parse_args()

train_df = data_loader.get_british_gp_data(years=(2021, 2022, 2023, 2024, 2025))
res = model_store.get_or_train_models(train_df, force_retrain=args.retrain)
print(f"Accuracy: {res['podium_accuracy']:.2f}")
print(f"Precision: {res['precision']:.2f}")
print(f"Recall: {res['recall']:.2f}")
print(f"ROC-AUC: {res['roc_auc']:.2f}")
print(f"MAE: {res['mae']:.2f}")

prediction_frame = data_loader.get_upcoming_british_gp_prediction()
if prediction_frame.empty:
    print("No 2026 prediction frame available.")
else:
    scored = []
    for _, row in prediction_frame.iterrows():
        podium_probability, predicted_position = model_logic.predict_record(res, row.to_dict())
        scored.append({
            "FullName": row.get("FullName", ""),
            "TeamName": row.get("TeamName", ""),
            "GridPosition": row.get("GridPosition", None),
            "Podium Probability": podium_probability,
            "Predicted Finish": round(predicted_position),
        })
    print(pd.DataFrame(scored).sort_values("Podium Probability", ascending=False).to_string(index=False))