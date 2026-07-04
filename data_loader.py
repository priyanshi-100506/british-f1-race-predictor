import os
import pandas as pd
import fastf1

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")


def _summarize_laps(session):
    if session is None:
        return {}
    laps = session.laps.copy().dropna(subset=["LapTime", "DriverNumber"])
    if laps.empty:
        return {}
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    summary = laps.groupby("DriverNumber")["LapTimeSeconds"].agg(["min", "mean", "count"])
    summary.columns = ["best_lap_time", "mean_lap_time", "lap_count"]
    return summary.to_dict("index")


def _weather_summary(session):
    if session is None or getattr(session, "weather_data", None) is None or session.weather_data.empty:
        return {}
    weather = session.weather_data
    return {
        "air_temp": float(weather["AirTemp"].mean()),
        "track_temp": float(weather["TrackTemp"].mean()),
        "humidity": float(weather["Humidity"].mean()),
        "rainfall": float(weather["Rainfall"].fillna(0).mean()),
    }


def _build_rows_from_results(results, year, event_name, practice_session=None, qualifying_session=None, race_session=None):
    rows = []

    if practice_session is None:
        try:
            practice_session = fastf1.get_session(year, event_name, "FP2")
            practice_session.load(telemetry=False)
        except Exception:
            practice_session = None

    if qualifying_session is None:
        try:
            qualifying_session = fastf1.get_session(year, event_name, "Q")
            qualifying_session.load(telemetry=False, weather=True)
        except Exception:
            qualifying_session = None

    if race_session is None:
        try:
            race_session = fastf1.get_session(year, event_name, "R")
            race_session.load(telemetry=False, weather=True)
        except Exception:
            race_session = None

    p_sum = _summarize_laps(practice_session)
    q_sum = _summarize_laps(qualifying_session)
    weather = _weather_summary(qualifying_session) or _weather_summary(race_session)

    for _, res in results.iterrows():
        if "DriverNumber" not in res.index:
            continue
        d_num = int(res["DriverNumber"])
        grid_position = res.get("GridPosition", res.get("Position", None))
        position = res.get("Position", None)
        row = {
            "FullName": res.get("FullName", ""),
            "Position": float(position) if pd.notna(position) else None,
            "GridPosition": float(grid_position) if pd.notna(grid_position) else None,
            "TeamName": res.get("TeamName", ""),
            "Year": year,
            "GP": event_name,
            "practice_best_lap_time": p_sum.get(d_num, {}).get("best_lap_time", None),
            "qualifying_best_lap_time": q_sum.get(d_num, {}).get("best_lap_time", None),
            "practice_mean_lap_time": p_sum.get(d_num, {}).get("mean_lap_time", None),
            "qualifying_mean_lap_time": q_sum.get(d_num, {}).get("mean_lap_time", None),
            "practice_lap_count": p_sum.get(d_num, {}).get("lap_count", None),
            "qualifying_lap_count": q_sum.get(d_num, {}).get("lap_count", None),
        }
        row.update(weather)
        rows.append(row)

    return rows


def _build_event_rows(year, event_name, session_name="R"):
    try:
        session = fastf1.get_session(year, event_name, session_name)
        session.load(telemetry=False, weather=True)
        practice = fastf1.get_session(year, event_name, "FP2")
        practice.load(telemetry=False)
        results = session.results.copy()
        return _build_rows_from_results(results, year, event_name, practice_session=practice, qualifying_session=None, race_session=session)
    except Exception:
        return []


def get_british_gp_data(years=(2021, 2022, 2023, 2024, 2025)):
    all_rows = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception:
            continue
        for _, event in schedule.iterrows():
            event_name = event.get("EventName", "")
            if "British" in event_name:
                all_rows.extend(_build_event_rows(year, event_name, session_name="R"))
    return pd.DataFrame(all_rows)


def get_upcoming_british_gp_prediction(year=2026, event_name="British Grand Prix", historical_data=None):
    if historical_data is None:
        historical_data = get_british_gp_data(years=(2021, 2022, 2023, 2024, 2025))

    try:
        qualifying = fastf1.get_session(year, event_name, "Q")
        qualifying.load(telemetry=False, weather=True)
        practice = fastf1.get_session(year, event_name, "FP2")
        practice.load(telemetry=False)
        results = qualifying.results.copy()
        return pd.DataFrame(_build_rows_from_results(results, year, event_name, practice_session=practice, qualifying_session=qualifying, race_session=qualifying))
    except Exception as exc:
        print(f"Unable to build a 2026 prediction frame from live session data: {exc}")

    if historical_data.empty:
        return pd.DataFrame()

    feature_columns = [
        "GridPosition",
        "practice_best_lap_time",
        "practice_mean_lap_time",
        "practice_lap_count",
        "qualifying_best_lap_time",
        "qualifying_mean_lap_time",
        "qualifying_lap_count",
        "air_temp",
        "track_temp",
        "humidity",
        "rainfall",
    ]
    driver_stats = historical_data.groupby("FullName", dropna=False).agg(
        TeamName=("TeamName", "last"),
        GridPosition=("GridPosition", "median"),
        practice_best_lap_time=("practice_best_lap_time", "median"),
        practice_mean_lap_time=("practice_mean_lap_time", "median"),
        practice_lap_count=("practice_lap_count", "median"),
        qualifying_best_lap_time=("qualifying_best_lap_time", "median"),
        qualifying_mean_lap_time=("qualifying_mean_lap_time", "median"),
        qualifying_lap_count=("qualifying_lap_count", "median"),
        air_temp=("air_temp", "median"),
        track_temp=("track_temp", "median"),
        humidity=("humidity", "median"),
        rainfall=("rainfall", "median"),
    ).reset_index()

    for col in feature_columns:
        if col not in driver_stats.columns:
            driver_stats[col] = historical_data[col].median() if col in historical_data.columns else None

    driver_stats = driver_stats.fillna({
        "GridPosition": historical_data["GridPosition"].median() if "GridPosition" in historical_data.columns else 10,
        "practice_best_lap_time": historical_data["practice_best_lap_time"].median() if "practice_best_lap_time" in historical_data.columns else 0,
        "practice_mean_lap_time": historical_data["practice_mean_lap_time"].median() if "practice_mean_lap_time" in historical_data.columns else 0,
        "practice_lap_count": historical_data["practice_lap_count"].median() if "practice_lap_count" in historical_data.columns else 0,
        "qualifying_best_lap_time": historical_data["qualifying_best_lap_time"].median() if "qualifying_best_lap_time" in historical_data.columns else 0,
        "qualifying_mean_lap_time": historical_data["qualifying_mean_lap_time"].median() if "qualifying_mean_lap_time" in historical_data.columns else 0,
        "qualifying_lap_count": historical_data["qualifying_lap_count"].median() if "qualifying_lap_count" in historical_data.columns else 0,
        "air_temp": historical_data["air_temp"].median() if "air_temp" in historical_data.columns else 25,
        "track_temp": historical_data["track_temp"].median() if "track_temp" in historical_data.columns else 30,
        "humidity": historical_data["humidity"].median() if "humidity" in historical_data.columns else 50,
        "rainfall": historical_data["rainfall"].median() if "rainfall" in historical_data.columns else 0,
    })

    driver_stats["Year"] = year
    driver_stats["GP"] = event_name
    return driver_stats[[
        "FullName",
        "TeamName",
        "GridPosition",
        "practice_best_lap_time",
        "practice_mean_lap_time",
        "practice_lap_count",
        "qualifying_best_lap_time",
        "qualifying_mean_lap_time",
        "qualifying_lap_count",
        "air_temp",
        "track_temp",
        "humidity",
        "rainfall",
        "Year",
        "GP",
    ]]