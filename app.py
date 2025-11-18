# app.py - Modern Pharma Sales Dashboard with AI Insights (Corporate Blue, Flat KPI Cards)
from groq import Groq
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime

# -----------------------
# Page config - must be first Streamlit command
# -----------------------
st.set_page_config(page_title="Pharma Sales Intelligence", layout="wide")

# -----------------------
# Load Groq client safely
# -----------------------
groq_client = None
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
except Exception:
    groq_client = None

# -----------------------
# Custom CSS (Corporate Blue + flat KPI cards)
# -----------------------
st.markdown(
    """
    <style>
    :root{
      --brand:#0b63b5;
      --muted:#6b778c;
      --card-bg:#ffffff;
      --page-bg:#f6f8fb;
    }
    body { background-color: var(--page-bg); }
    .header {
      background: linear-gradient(90deg, var(--brand), #0d74d1);
      padding: 18px 24px;
      border-radius: 8px;
      color: white;
      margin-bottom: 18px;
    }
    .kpi {
      background: var(--card-bg);
      border: 1px solid #e6eef9;
      padding: 14px;
      border-radius: 8px;
      box-shadow: none;
    }
    .kpi .label { color: var(--muted); font-size:12px; }
    .kpi .value { font-size:20px; font-weight:700; color:#0b63b5; margin-top:6px; }
    .section-title { font-size:18px; font-weight:700; color:#0b63b5; margin-top:8px; }
    .ai-box { background: white; border: 1px solid #e6eef9; padding:12px; border-radius:8px; }
    .small-muted { color:var(--muted); font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Header
# -----------------------
st.markdown(
    f"""
    <div class="header">
      <h2 style="margin:0;">Pharma Sales Intelligence Dashboard</h2>
      <div style="font-size:13px; margin-top:6px;">Forecasting, insights and AI-driven recommendations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Sidebar - Upload + Product + AI Buttons
# -----------------------
st.sidebar.header("1) Upload Data")
daily = st.sidebar.file_uploader("Daily Sales CSV", type=["csv"])
weekly = st.sidebar.file_uploader("Weekly Sales CSV", type=["csv"])
monthly = st.sidebar.file_uploader("Monthly Sales CSV", type=["csv"])
hourly = st.sidebar.file_uploader("Hourly Sales CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("2) Product & Options")

# appearance toggle
show_kpis = st.sidebar.checkbox("Show KPI cards", value=True)
show_charts = st.sidebar.checkbox("Show Charts", value=True)

# AI buttons area
st.sidebar.markdown("---")
st.sidebar.header("3) AI Insights (press a button)")

btn_explain_perf = st.sidebar.button("AI: Explain Forecast Performance")
btn_exec_summary = st.sidebar.button("AI: Generate Executive Summary")
btn_anomalies = st.sidebar.button("AI: Detect Anomalies")
btn_marketing = st.sidebar.button("AI: Recommend Marketing Strategy")
btn_compare = st.sidebar.button("AI: Compare Top Products")
btn_seasonality = st.sidebar.button("AI: Explain Seasonal Patterns")
btn_drivers = st.sidebar.button("AI: Sales Drivers & Reasoning")
btn_pricing = st.sidebar.button("AI: Pricing Recommendation")
btn_risk = st.sidebar.button("AI: Risk Analysis")
btn_customer = st.sidebar.button("AI: Customer Behavior Insights")
btn_opportunity = st.sidebar.button("AI: Opportunity Detection")
# Chat input (simple)
st.sidebar.markdown("---")
st.sidebar.subheader("Ask AI (quick)")
chat_query = st.sidebar.text_input("Ask a question about the data and forecasts")

# -----------------------
# Validate uploads
# -----------------------
if not (daily and weekly and monthly and hourly) and not chat_query:
    st.sidebar.info("Upload 4 CSV files (daily, weekly, monthly, hourly) to enable full features.")
if not (daily and weekly and monthly and hourly):
    st.stop()

# -----------------------
# Read & preprocess uploaded CSVs
# -----------------------
def safe_read_csv(f):
    try:
        return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()

daily = safe_read_csv(daily)
weekly = safe_read_csv(weekly)
monthly = safe_read_csv(monthly)
hourly = safe_read_csv(hourly)

for df_temp, freq in [(daily, "daily"), (weekly, "weekly"), (monthly, "monthly"), (hourly, "hourly")]:
    if df_temp is not None and len(df_temp) > 0:
        df_temp.columns = df_temp.columns.str.lower()

# add frequency column
if "frequency" not in daily.columns:
    daily["frequency"] = "daily"
if "frequency" not in weekly.columns:
    weekly["frequency"] = "weekly"
if "frequency" not in monthly.columns:
    monthly["frequency"] = "monthly"
if "frequency" not in hourly.columns:
    hourly["frequency"] = "hourly"

# combine
df = pd.concat([daily, weekly, monthly, hourly], ignore_index=True)
df.columns = df.columns.str.lower()

# ensure date column named 'datum'
if "date" in df.columns and "datum" not in df.columns:
    df.rename(columns={"date": "datum"}, inplace=True)
if "datum" not in df.columns:
    # try common names
    if "ds" in df.columns:
        df.rename(columns={"ds": "datum"}, inplace=True)
    elif "timestamp" in df.columns:
        df.rename(columns={"timestamp": "datum"}, inplace=True)

# coerce date
df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
df = df.dropna(subset=["datum"])

# identify id vars and product cols
id_vars = [c for c in ["datum", "year", "month", "hour", "weekday name", "frequency"] if c in df.columns]
product_cols = [c for c in df.columns if c not in id_vars]

# melt to long
df_long = df.melt(id_vars=id_vars, value_vars=product_cols, var_name="product", value_name="sales")
df_long["product"] = df_long["product"].astype(str).str.upper()
df_long["sales"] = pd.to_numeric(df_long["sales"], errors="coerce")
df_long = df_long.dropna(subset=["sales"])
df_long = df_long[df_long["sales"] > 0]

# remove outliers (IQR)
if len(df_long) > 10:
    Q1, Q3 = df_long["sales"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_long = df_long[(df_long["sales"] >= Q1 - 1.5 * IQR) & (df_long["sales"] <= Q3 + 1.5 * IQR)]

# top products
top_products = df_long.groupby("product")["sales"].sum().sort_values(ascending=False)

st.sidebar.subheader("Select Product")
selected_product = st.sidebar.selectbox("Product", top_products.index)

# -----------------------
# Helper: call Groq LLM safely and return text
# -----------------------
def ask_ai(prompt, model="llama-3.3-70b-versatile", max_tokens=700, temperature=0.2):
    if groq_client is None:
        return "Groq API key not configured. Please set GROQ_API_KEY in Streamlit secrets or environment."
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful, business-focused data analyst."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI call failed: {str(e)}"

# -----------------------
# Forecasting for selected product
# -----------------------
prod = df_long[df_long["product"] == selected_product].groupby("datum")["sales"].sum().reset_index()
prod = prod.rename(columns={"datum": "ds", "sales": "y"}).sort_values("ds")
if len(prod) < 20:
    st.error("Not enough data for this product to model. Choose another product or upload more data.")
    st.stop()

# remove product-level outliers
Q1, Q3 = prod["y"].quantile([0.25, 0.75])
IQR = Q3 - Q1
prod = prod[(prod["y"] >= Q1 - 1.5 * IQR) & (prod["y"] <= Q3 + 1.5 * IQR)]

split_index = int(len(prod) * 0.80)
train = prod.iloc[:split_index]
test = prod.iloc[split_index:]

# fit Prophet
model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model.fit(train)

# predict test
future_test = test[["ds"]]
forecast_test = model.predict(future_test)[["ds", "yhat"]]
test_eval = test.merge(forecast_test, on="ds", how="left").dropna(subset=["yhat"])
test_eval = test_eval.rename(columns={"yhat": "y_pred"})
if len(test_eval) == 0:
    st.error("Model failed to predict. Try different product.")
    st.stop()

# metrics
mae = mean_absolute_error(test_eval["y"], test_eval["y_pred"])
mse = mean_squared_error(test_eval["y"], test_eval["y_pred"])
rmse = np.sqrt(mse)
r2 = r2_score(test_eval["y"], test_eval["y_pred"])

# full future forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
forecast_next = forecast.tail(30).copy()
forecast_next["ds_fmt"] = forecast_next["ds"].dt.strftime("%d-%b")

# seasonal components
weekly_comp = model.predict_seasonal_components(pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=7)}))
weekly_comp["weekly"] = weekly_comp["weekly"].clip(lower=0)
trend_df = forecast[["ds", "trend"]].copy()
trend_df["trend"] = trend_df["trend"].clip(lower=0)

# -----------------------
# KPI cards (Simple Flat Cards - style A)
# -----------------------
if show_kpis:
    total_sales = int(df_long["sales"].sum())
    avg_daily = float(prod["y"].mean())
    last_actual = int(prod["y"].iloc[-1])
    growth = None
    if len(prod) >= 2:
        growth = (prod["y"].iloc[-1] - prod["y"].iloc[0]) / (prod["y"].iloc[0] + 1e-9) * 100
        growth = round(growth, 1)
    else:
        growth = 0.0

    k1, k2, k3, k4 = st.columns([1.8,1.4,1.4,1.4])
    with k1:
        st.markdown('<div class="kpi"><div class="label">Total Sales (all products)</div>'
                    f'<div class="value">{total_sales:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi"><div class="label">Avg Sales (selected)</div>'
                    f'<div class="value">{avg_daily:.1f}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi"><div class="label">Last Actual Sales (selected)</div>'
                    f'<div class="value">{last_actual}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi"><div class="label">Growth % (first→last)</div>'
                    f'<div class="value">{growth}%</div></div>', unsafe_allow_html=True)

# -----------------------
# Main content: Charts + Results
# -----------------------
st.markdown('<div class="section-title">Forecast & Charts</div>', unsafe_allow_html=True)

# metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", round(mae,2))
col2.metric("MSE", round(mse,2))
col3.metric("RMSE", round(rmse,2))
col4.metric("R²", round(r2,3))

if show_charts:
    # actual vs predicted (bar)
    st.subheader("Average Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(["Actual","Predicted"], [test_eval["y"].mean(), test_eval["y_pred"].mean()], color=["#0b63b5","#0d74d1"])
    ax.set_ylabel("Average Sales")
    st.pyplot(fig)

    # next 30 days forecast
    st.subheader("Next 30 Days Forecast")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.bar(forecast_next["ds_fmt"], forecast_next["yhat"], color="#0d74d1")
    ax2.set_xticklabels(forecast_next["ds_fmt"], rotation=90)
    st.pyplot(fig2)

    # seasonality
    st.subheader("Weekly Seasonality")
    figw, axw = plt.subplots(figsize=(8,3))
    axw.plot(["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], weekly_comp["weekly"], marker='o')
    st.pyplot(figw)

    # trend
    st.subheader("Trend Over Time")
    figt, axt = plt.subplots(figsize=(10,3))
    axt.plot(trend_df["ds"], trend_df["trend"], linewidth=2)
    axt.xaxis.set_major_locator(plt.MaxNLocator(10))
    axt.tick_params(axis='x', rotation=45)

    axt.tick_params(axis='x', rotation=45)
    st.pyplot(figt)

# -----------------------
# AI Actions (Triggered by sidebar buttons)
# Results will display in main area below charts
# -----------------------
st.markdown('<div class="section-title">AI Insights</div>', unsafe_allow_html=True)

# helper to build a dataset snippet for prompts
def dataset_sample_text(n=6):
    try:
        return df.head(n).to_string(index=False)
    except Exception:
        return "No data sample available."

# 1) Explain Forecast Performance (simple client language)
if btn_explain_perf:
    prompt = f"""
Explain the forecasting results below in very simple, client-friendly language.
Do NOT use technical jargon or metric names. Keep it short and practical.

Forecast metrics:
- MAE: {mae:.2f}
- RMSE: {rmse:.2f}
- R²: {r2:.3f}

Explain:
1) What it means when predictions are close to actual sales.
2) Why predictions sometimes differ from real sales.
3) How a business manager should interpret these numbers and what actions to take.

Data sample:
{dataset_sample_text(5)}
"""
    with st.spinner("AI is preparing a simple explanation..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Forecast Explanation (Client-Friendly)")
    st.write(text)

# 2) Executive Summary
if btn_exec_summary:
    prompt = f"""
Create a short executive summary (3-5 bullets) for management from this sales data and forecast.
Write in plain English, focus on insights and recommended actions.

Data sample:
{dataset_sample_text(5)}
"""
    with st.spinner("AI is generating executive summary..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Executive Summary")
    st.write(text)

# 3) Anomaly Detection
if btn_anomalies:
    snippet = df_long.sort_values(["product","sales"], ascending=[True, False]).head(40).to_string(index=False)
    prompt = f"""
Scan this sales data snippet and list any likely anomalies: sudden spikes or drops, repeated abnormal values, or suspicious entries.
Provide dates (if visible) and simple possible reasons (data error, promotion, seasonality).
Data snippet:
{snippet}
"""
    with st.spinner("AI is detecting anomalies..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Anomalies & Possible Reasons")
    st.write(text)

# 4) Marketing Strategy
if btn_marketing:
    prompt = f"""
Based on this sales data, propose a short marketing plan to increase sales for '{selected_product}'.
Be practical: suggest channels, timing (weeks/months), and one simple campaign idea.
Data sample:
{dataset_sample_text(5)}
"""
    with st.spinner("AI is suggesting marketing actions..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Marketing Recommendations")
    st.write(text)

# 5) Compare Top Products
if btn_compare:
    top3 = list(top_products.head(3).index)
    prompt = f"""
Compare these top 3 products: {top3}. Explain in plain business terms which is doing best and why, and which shows the best growth potential.
Include simple bullets describing strengths and weaknesses.
Data sample:
{dataset_sample_text(5)}
"""
    with st.spinner("AI is comparing products..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Top Product Comparison")
    st.write(text)

# 6) Explain Seasonal Patterns
if btn_seasonality:
    weekly_vals = weekly_comp["weekly"].tolist()
    prompt = f"""
Explain the weekly seasonal pattern represented by these 7 numbers (Sun→Sat):
{weekly_vals}
Explain in simple terms what business activities or customer behaviors might cause this pattern.
"""
    with st.spinner("AI is explaining seasonality..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Seasonality Explanation")
    st.write(text)

# 7) Sales Drivers & Reasoning
if btn_drivers:
    prompt = f"""
Explain likely drivers of sales for '{selected_product}' given this data. Consider seasonality, promotions, and customer behavior.
Give simple bullet points and recommended checks (e.g., check promo calendar, distribution).
Data sample:
{dataset_sample_text(6)}
"""
    with st.spinner("AI is identifying drivers..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Sales Drivers & Recommendations")
    st.write(text)

# 8) Pricing Recommendation
if btn_pricing:
    prompt = f"""
Provide a simple pricing recommendation for '{selected_product}' to maximize revenue without hurting demand. Suggest 1 conservative action and 1 moderate action.
Keep it short and business-focused.
"""
    with st.spinner("AI is suggesting pricing actions..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Pricing Recommendations")
    st.write(text)

# 9) Risk Analysis
if btn_risk:
    prompt = f"""
Identify top business risks visible from the dataset (e.g., stockouts, demand drops, seasonality exposure). Provide short mitigation steps for each risk.
Data sample:
{dataset_sample_text(6)}
"""
    with st.spinner("AI is performing risk analysis..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Risk Analysis & Mitigations")
    st.write(text)

# 10) Customer Behavior Insights
if btn_customer:
    prompt = f"""
Explain likely customer behavior patterns seen in the sales data. Use plain language and suggest one action to engage customers better.
Data sample:
{dataset_sample_text(6)}
"""
    with st.spinner("AI is analyzing customer behavior..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Customer Behavior Insights")
    st.write(text)

# 11) Opportunity Detection
if btn_opportunity:
    prompt = f"""
Identify 2-3 growth opportunities or product expansion ideas based on sales trends. Keep suggestions practical with a short rationale.
Data sample:
{dataset_sample_text(6)}
"""
    with st.spinner("AI is finding opportunities..."):
        text = ask_ai(prompt)
        text = text.replace("<hr>", "").replace("<hr/>", "").replace("<hr />", "")

    st.subheader("Opportunity Detection")
    st.write(text)

# 12) Natural Language Chat (quick)
if chat_query and len(chat_query.strip()) > 0:
    prompt = f"""
You are a helpful business analyst. Answer the user's question about the sales data and forecasts clearly and concisely.

User question:
{chat_query}

Context - sample data:
{dataset_sample_text(6)}
"""
    with st.spinner("AI is answering your question..."):
        text = ask_ai(prompt, max_tokens=600)
    st.subheader("AI Chat Response")
    st.write(text)

# -----------------------
# Footer / end
# -----------------------
st.success("Dashboard Loaded Successfully.")
