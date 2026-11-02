from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from model_deployment import load_history, load_model, load_metadata, forecast_hours

st.set_page_config(
    page_title="PJM Energy Forecast Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#0b1f3a,#164e63); color: white; margin-bottom: 1rem;}
    .hero h1 {font-size: 2.15rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {font-size: 1rem; opacity: .9; margin: 0;}
    .note {padding: .8rem 1rem; border-left: 4px solid #e0a526; background: rgba(224,165,38,.10); border-radius: 6px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); padding: 14px; border-radius: 12px;}
</style>
<div class="hero">
  <h1>⚡ PJM Energy Forecast Studio</h1>
  <p>Hourly electricity demand forecasting · XGBoost · Forecasts in megawatts (MW)</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_history():
    return load_history()

@st.cache_data
def get_metadata():
    return load_metadata()

try:
    history = get_history()
    model = get_model()
    metadata = get_metadata()
except Exception as exc:
    st.error(f"Could not load the forecasting assets: {exc}")
    st.info("Keep app.py, model_deployment.py, the .joblib model, metadata JSON, and cleaned history CSV together.")
    st.stop()

first_date = history.index.min().date()
last_date = history.index.max().date()

with st.sidebar:
    st.header("Forecast controls")
    horizon_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=30, step=1)
    run_forecast = st.button("Generate forecast", type="primary", use_container_width=True)
    st.divider()
    st.caption("Model")
    st.write(f"{metadata.get('selected_model', 'Selected model')} Regressor")
    st.caption("Frequency")
    st.write("Hourly")
    st.caption("Target")
    st.write("PJMW_MW")

st.markdown(
    '<div class="note"><b>Important data limitation:</b> the supplied PJMW historical dataset ends on '
    f'{history.index[-1]:%d %b %Y, %H:%M}. Forecasts continue from that historical endpoint. '
    'They are a project demonstration, not live demand predictions for today.</div>',
    unsafe_allow_html=True,
)
st.write("")

overview_tab, forecast_tab, model_tab = st.tabs(["📊 Overview", "🔮 Forecast & download", "🧠 Model details"])

with overview_tab:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Historical observations", f"{len(history):,}")
    c2.metric("History begins", history.index.min().strftime("%d %b %Y"))
    c3.metric("History ends", history.index.max().strftime("%d %b %Y"))
    c4.metric("Latest recorded demand", f"{history.iloc[-1]:,.0f} MW")

    st.subheader("Historical energy consumption")
    hist_days = st.selectbox(
        "History shown", [30, 90, 180, 365, 0], index=2,
        format_func=lambda x: "All available history" if x == 0 else f"Last {x} days",
        key="preset_history_days"
    )
    use_custom_dates = st.checkbox("Choose a custom historical date range", value=False)
    if use_custom_dates:
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start date", value=max(first_date, last_date - pd.Timedelta(days=30)),
                min_value=first_date, max_value=last_date, key="history_start_date"
            )
        with date_col2:
            end_date = st.date_input(
                "End date", value=last_date, min_value=first_date, max_value=last_date,
                key="history_end_date"
            )
        if start_date > end_date:
            st.error("Start date must be on or before the end date.")
            visible_history = history.iloc[0:0]
        else:
            start_ts = pd.Timestamp(start_date)
            # Include the entire selected end date, through its final recorded hour.
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            visible_history = history.loc[(history.index >= start_ts) & (history.index <= end_ts)]
        period_label = f"{start_date} to {end_date}"
    else:
        visible_history = history if hist_days == 0 else history.loc[
            history.index >= history.index[-1] - pd.Timedelta(days=hist_days)
        ]
        period_label = "All available history" if hist_days == 0 else f"Last {hist_days} days"

    if visible_history.empty:
        st.warning("No historical observations were found for the selected range.")
    else:
        hist_df = visible_history.rename("Actual MW").rename_axis("Datetime").reset_index()
        fig = px.line(
            hist_df, x="Datetime", y="Actual MW",
            title=f"Observed hourly PJMW demand — {period_label}",
            labels={"Datetime":"Timestamp", "Actual MW":"Consumption (MW)"}
        )
        fig.update_layout(hovermode="x unified", margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historical readings for selected period")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Readings", f"{len(visible_history):,}")
        metric2.metric("Average", f"{visible_history.mean():,.2f} MW")
        metric3.metric("Minimum", f"{visible_history.min():,.2f} MW")
        metric4.metric("Maximum", f"{visible_history.max():,.2f} MW")

        historical_table = visible_history.rename("Actual_MW").rename_axis("Datetime").reset_index()
        historical_table["Date"] = historical_table["Datetime"].dt.strftime("%Y-%m-%d")
        historical_table["Time"] = historical_table["Datetime"].dt.strftime("%H:%M")
        historical_table = historical_table[["Datetime", "Date", "Time", "Actual_MW"]]
        st.dataframe(
            historical_table.rename(columns={"Datetime":"Timestamp", "Actual_MW":"Actual consumption (MW)"}).round(2),
            use_container_width=True, hide_index=True, height=350
        )
        hist_csv = historical_table[["Datetime", "Actual_MW"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download selected historical readings (CSV)", data=hist_csv,
            file_name="PJM_selected_historical_readings.csv", mime="text/csv",
            key="download_historical"
        )
        st.caption("The end date includes all available hourly readings recorded on that date.")
    st.caption("Use this view to inspect historical demand patterns. These are actual values from the supplied dataset.")

with forecast_tab:
    st.subheader("Forecast future hourly demand")
    st.write(
        f"Generate **{horizon_days} day(s)** of hourly predictions. The app forecasts recursively, "
        "using each predicted hour to build features for the next hour."
    )
    st.info(
        f"Forecast origin: last available observation at **{history.index[-1]:%d %b %Y, %H:%M}**. "
        f"For a {horizon_days}-day horizon, predictions end at "
        f"**{(history.index[-1] + pd.Timedelta(hours=horizon_days*24)):%d %b %Y, %H:%M}**."
    )
    total_hours = horizon_days * 24
    if run_forecast or "last_forecast_hours" not in st.session_state:
        try:
            with st.spinner("Generating hourly predictions..."):
                result = forecast_hours(model, history, total_hours)
            st.session_state["forecast_result"] = result
            st.session_state["last_forecast_hours"] = total_hours
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            result = None
    else:
        result = st.session_state.get("forecast_result")
        if result is not None and len(result) != total_hours:
            result = None

    if result is not None:
        result = result.copy()
        daily = result.set_index("Datetime")["Predicted_MW"].resample("D").agg(["mean","min","max"]).reset_index()
        a,b,c,d = st.columns(4)
        a.metric("Hourly predictions", f"{len(result):,}")
        b.metric("Average forecast", f"{result.Predicted_MW.mean():,.0f} MW")
        c.metric("Peak forecast", f"{result.Predicted_MW.max():,.0f} MW")
        d.metric("Lowest forecast", f"{result.Predicted_MW.min():,.0f} MW")
        st.subheader("Forecast curve")
        chart = px.line(
            result, x="Datetime", y="Predicted_MW",
            labels={"Datetime":"Forecast timestamp","Predicted_MW":"Predicted demand (MW)"},
            title=f"Next {horizon_days} day(s) — hourly forecast"
        )
        chart.update_layout(hovermode="x unified", margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(chart, use_container_width=True)
        st.subheader("Daily summary")
        daily = daily.rename(columns={"mean":"Average MW","min":"Minimum MW","max":"Maximum MW"})
        st.dataframe(daily.round(2), use_container_width=True, hide_index=True)
        st.subheader("Hourly predictions")
        st.dataframe(
            result[["Datetime","Predicted_MW"]].rename(
                columns={"Datetime":"Forecast timestamp","Predicted_MW":"Predicted consumption (MW)"}
            ).round(2),
            use_container_width=True, height=340, hide_index=True
        )
        csv = result[["Datetime","Predicted_MW"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download hourly forecast (CSV)", data=csv,
            file_name=f"PJM_hourly_forecast_{horizon_days}_days.csv",
            mime="text/csv", type="primary"
        )
        st.caption("Predictions are point estimates. No prediction intervals are supplied.")
    else:
        st.info("Click **Generate forecast** in the sidebar to refresh predictions for the selected horizon.")

with model_tab:
    st.subheader("Deployment model and evaluation")
    left,right = st.columns(2)
    with left:
        st.markdown("**Model selected during notebook evaluation**")
        st.write(metadata.get("selected_model", "XGBoost"))
        st.markdown("**Training design**")
        st.write(f"Training window: approximately {metadata.get('training_window_days',365)} days")
        st.write(f"Rows used for fitting: {metadata.get('training_rows','—'):,}")
        st.write("Recursive one-hour-at-a-time predictions")
        st.write("Calendar cycles, lag values, and rolling summaries")
    with right:
        st.markdown("**Held-out test metrics reported in your notebook**")
        metrics = metadata.get("heldout_test_metrics_from_notebook", {})
        metric_labels = [("MAE_MW","MAE (MW)"),("RMSE_MW","RMSE (MW)"),
                         ("MAPE_percent","MAPE (%)"),("R2","R²")]
        for key,label in metric_labels:
            if key in metrics:
                st.metric(label, f"{metrics[key]:,.3f}")
        st.caption("Metrics are from the notebook's historical test evaluation; they are not a guarantee for future data.")
    st.markdown("**Selected configuration**")
    st.json(metadata.get("selected_configuration", {}))
    st.markdown("**Feature inputs used by the model**")
    st.write(", ".join(metadata.get("feature_columns", [])))
    st.markdown("**Interpretation**")
    st.write("MAE and RMSE are measured in MW; lower is better. MAPE is a percentage error measure. R² describes variance explained on the evaluated test sample and is not a prediction-accuracy percentage.")
    st.markdown("**Operational limitations**")
    st.write("- This app uses the provided historical file, which ends in August 2018.")
    st.write("- Results depend on the quality and time coverage of the training data.")
    st.write("- Refresh the data and retrain/revalidate the model before using it for current operational decisions.")
