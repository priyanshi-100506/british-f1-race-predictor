import pandas as pd
import streamlit as st
import data_loader
import model_logic

st.set_page_config(page_title="British GP Predictor", page_icon="🏎️", layout="wide")
st.title("🇬🇧 British GP Race Predictor")


@st.cache_data(show_spinner=False)
def load_dataset():
    return data_loader.get_british_gp_data()


with st.spinner("Fetching British GP history from FastF1..."):
    results = load_dataset()

if results.empty:
    st.error("No British GP rows were loaded. Please check the FastF1 connection or cached data.")
    st.stop()

with st.spinner("Training the podium and finishing-position models..."):
    model_bundle = model_logic.train_british_gp_model(results)

st.subheader("Model performance")
col1, col2 = st.columns(2)
col1.metric("Podium accuracy", f"{model_bundle['podium_accuracy']:.2%}")
col2.metric("Position RMSE", f"{model_bundle['position_rmse']:.2f}")

st.subheader("2026 British GP podium prediction")
prediction_frame = data_loader.get_upcoming_british_gp_prediction()
if prediction_frame.empty:
    st.info("No 2026 qualifying data was available for prediction yet.")
else:
    scored_rows = []
    for _, row in prediction_frame.iterrows():
        podium_probability, predicted_position = model_logic.predict_record(model_bundle, row.to_dict())
        scored_rows.append({
            "FullName": row.get("FullName", ""),
            "TeamName": row.get("TeamName", ""),
            "GridPosition": row.get("GridPosition", None),
            "Podium Probability": podium_probability,
            "Predicted Finish": round(predicted_position),
        })

    scored_df = pd.DataFrame(scored_rows).sort_values("Podium Probability", ascending=False)
    st.dataframe(scored_df.head(20))

st.subheader("Historical sample")
show_cols = [col for col in ["FullName", "Position", "GridPosition", "TeamName", "Year", "GP"] if col in results.columns]
st.dataframe(results[show_cols].head(20))