import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

try:
    import secrets
except Exception:
    pass

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

groq_client = None 

if GROQ_AVAILABLE:
    try:
        groq_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        groq_key = None
        
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    
    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
        except Exception:
            groq_client = None


st.set_page_config(page_title="Trend Analyst and Forecast", layout="wide")

st.markdown("""
<style>
:root{--brand:#0b63b5;--muted:#6b778c;--card:#ffffff;--page:#f6f8fb;}
body { background-color: var(--page); }
.header {
    background: linear-gradient(90deg,var(--brand),#0d74d1);
    padding:16px; border-radius:8px; color:#fff; margin-bottom:12px;
}
.small-muted { color:var(--muted); font-size:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header"><h2 style="margin:0">Trend Analyst and Forecast</h2></div>',
    unsafe_allow_html=True
)

st.sidebar.header("1 — Upload CSV file(s)")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more CSV files",
    accept_multiple_files=True,
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.header("2 — Analysis options")

forecast_days = st.sidebar.number_input("Days to forecast", 1, 365, 30)
recent_days = st.sidebar.number_input("Recent days for moving chart", 7, 365, 90)
rolling_window = st.sidebar.number_input("Moving window (days)", 1, 90, 7)

enable_forecast = st.sidebar.checkbox("Enable forecasting", value=PROPHET_AVAILABLE)
manual_override = st.sidebar.checkbox("Manual column selection")

st.sidebar.markdown("---")
st.sidebar.header("3 — AI Insight Options")

ai_explain_forecast = st.sidebar.checkbox("AI: Explain forecast performance")
ai_exec_summary = st.sidebar.checkbox("AI: Executive summary")
ai_detect_anomalies = st.sidebar.checkbox("AI: Detect anomalies")
ai_marketing = st.sidebar.checkbox("AI: Marketing suggestion")
ai_compare = st.sidebar.checkbox("AI: Compare top products")
ai_seasonality = st.sidebar.checkbox("AI: Explain weekly pattern")

st.sidebar.markdown("---")
st.sidebar.header("Ask AI Anything")
custom_ai_question = st.sidebar.text_area("Type your question", height=80)
run_custom_ai = st.sidebar.button("Ask AI")

st.sidebar.markdown("---")
run_ai_now = st.sidebar.button("Run AI Insights")

def safe_read_csv(f):
    try:
        return pd.read_csv(f)
    except:
        try:
            return pd.read_csv(f, encoding="latin-1")
        except:
            return pd.DataFrame()

def detect_date_col(df):
    possible = ["date", "ds", "timestamp", "time", "datetime"]
    for c in possible:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= 3:
                return c
    best = None; best_valid = -1
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): 
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        valid = parsed.notna().sum()
        if valid > best_valid:
            best_valid = valid
            best = c
    return best if best_valid >= 3 else None

def detect_numeric_col(df):
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not nums:
        return None
    scores = {c: (df[c].notna().sum(), df[c].nunique()) for c in nums}
    return max(scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]

def set_date_ticks(ax, labels, max_ticks=8):
    if not labels:
        return
    step = max(1, len(labels)//max_ticks)
    ticks = list(range(0, len(labels), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=45, ha='right')

def ask_ai(prompt, max_tokens=400):
    if groq_client is None:
        return "AI not configured. Please check GROQ_API_KEY setup."
    try:
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role":"system","content":"Explain in simple plain English."},
                {"role":"user","content":prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"


if not uploaded_files:
    st.info("Please upload at least one CSV file.")
    st.stop()

frames = []
for f in uploaded_files:
    df_temp = safe_read_csv(f)
    if df_temp is None or df_temp.empty:
        st.warning(f"File unreadable — skipped.")
        continue
    df_temp.columns = [str(c).strip().lower() for c in df_temp.columns]  # normalize
    frames.append(df_temp)

if not frames:
    st.error("No usable CSV data.")
    st.stop()

df_all = pd.concat(frames, ignore_index=True, sort=False)

st.markdown("<div class='small-muted'>Columns detected: " + ", ".join(df_all.columns[:25]) + "</div>", unsafe_allow_html=True)

auto_date = detect_date_col(df_all)
auto_value = detect_numeric_col(df_all)

if manual_override:
    date_col = st.selectbox("Choose date column", [None] + list(df_all.columns),
        index=(list(df_all.columns).index(auto_date) + 1) if auto_date in df_all.columns else 0)

    numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    value_col = st.selectbox("Choose numeric column", [None] + numeric_cols,
        index=(numeric_cols.index(auto_value) + 1) if auto_value in numeric_cols else 0)
else:
    date_col = auto_date
    value_col = auto_value

st.markdown(
    f"<div class='small-muted'>Using date: <b>{date_col}</b> — value: <b>{value_col}</b></div>",
    unsafe_allow_html=True
)

if date_col is None or value_col is None:
    st.error("Could not detect a valid date or numeric column.")
    st.stop()

df_all[date_col] = pd.to_datetime(df_all[date_col], errors="coerce")
df_all[value_col] = pd.to_numeric(df_all[value_col], errors="coerce")

df_clean = df_all.dropna(subset=[date_col, value_col]).copy()
if df_clean.shape[0] < 10:
    st.error("Not enough valid rows after cleaning.")
    st.stop()
non_numeric_cols = [
    c for c in df_clean.columns
    if c not in [date_col, value_col] and not pd.api.types.is_numeric_dtype(df_clean[c])
]

selected_cat = None
selected_cat_val = None

if non_numeric_cols:
    use_cat = st.selectbox("Optional: filter by category", ["None"] + non_numeric_cols)
    if use_cat != "None":
        selected_cat = use_cat
        vals = sorted(df_clean[selected_cat].dropna().astype(str).unique())
        selected_cat_val = st.selectbox(f"Choose value for {selected_cat}", vals)

if selected_cat and selected_cat_val:
    df_focus = df_clean[df_clean[selected_cat].astype(str) == str(selected_cat_val)].copy()
else:
    df_focus = df_clean.copy()


series = (
    df_focus.groupby(date_col)[value_col]
    .sum()
    .reset_index()
    .rename(columns={date_col: "ds", value_col: "y"})
    .sort_values("ds")
    .reset_index(drop=True)
)

series_pos = series.copy()
if (series_pos["y"] <= 0).all():
    st.error("All values are zero/negative — cannot analyze.")
    st.stop()

series_pos["y"] = series_pos["y"].clip(lower=1.0)


st.subheader("Quick summary")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total days available", len(series_pos))
with col2:
    st.metric("Time range", f"{series_pos['ds'].min().date()} → {series_pos['ds'].max().date()}")
with col3:
    st.metric("Average daily value", f"{series_pos['y'].mean():.2f}")


st.subheader(f"{rolling_window}-day moving average")

series_pos["rolling"] = series_pos["y"].rolling(
    window=int(rolling_window), min_periods=1
).mean()

recent = series_pos.tail(int(recent_days)).copy()
labels_recent = recent["ds"].dt.strftime("%Y-%m-%d").tolist()

fig_r, ax_r = plt.subplots(figsize=(10, 3))
ax_r.bar(range(len(recent)), recent["rolling"], color="#0b63b5")
ax_r.set_ylabel("Average value")
ax_r.set_title(f"Recent {recent_days} days")

set_date_ticks(ax_r, labels_recent)
plt.tight_layout()
st.pyplot(fig_r)

st.subheader("Monthly totals")

series_pos["month"] = series_pos["ds"].dt.to_period("M").astype(str)
monthly = series_pos.groupby("month")["y"].sum().reset_index()
show_months = monthly.tail(24)

fig_m, ax_m = plt.subplots(figsize=(10, 3))
ax_m.bar(range(len(show_months)), show_months["y"], color="#0d74d1")
ax_m.set_ylabel("Total value")
ax_m.set_title("Last 24 months")

set_date_ticks(ax_m, show_months["month"].tolist())
plt.tight_layout()
st.pyplot(fig_m)

st.subheader("Yearly totals")

series_pos["year"] = series_pos["ds"].dt.year
yearly = series_pos.groupby("year")["y"].sum().reset_index()

if len(yearly) <= 1:     # fallback
    show_data = monthly.tail(12)
    labels = show_data["month"].tolist()
    values = show_data["y"].tolist()
    title = "Last 12 Months (fallback)"
else:
    show_data = yearly
    labels = yearly["year"].astype(str).tolist()
    values = yearly["y"].tolist()
    title = "Yearly totals"

fig_y, ax_y = plt.subplots(figsize=(8, 3))
ax_y.bar(labels, values, color="#1d88e5")
ax_y.set_ylabel("Total value")
ax_y.set_title(title)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_y)

st.subheader("Day-of-week pattern")

series_pos["weekday"] = series_pos["ds"].dt.day_name()
week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

weekly = series_pos.groupby("weekday")["y"].mean().reindex(week_order)

fig_w, ax_w = plt.subplots(figsize=(8, 3))
ax_w.bar(week_order, weekly, color="#0b63b5")
ax_w.set_ylabel("Average value")
ax_w.set_title("Average value by weekday")

plt.tight_layout()
st.pyplot(fig_w)


st.subheader("Quarterly totals")

series_pos["quarter_clean"] = (
    series_pos["ds"].dt.year.astype(str) +
    " Q" +
    series_pos["ds"].dt.quarter.astype(str)
)

quarterly = (
    series_pos.groupby("quarter_clean")["y"]
    .sum()
    .reset_index()
)

quarterly = quarterly.tail(20)

fig_q, ax_q = plt.subplots(figsize=(10, 3))
ax_q.bar(quarterly["quarter_clean"], quarterly["y"], color="#147ae0")
ax_q.set_ylabel("Total value")
ax_q.set_title("Quarterly totals")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig_q)

st.markdown("---")
st.subheader("Forecast & simple accuracy")

mae = rmse = r2 = mape = None

if enable_forecast and not PROPHET_AVAILABLE:
    st.info("Prophet is not installed. Install prophet to enable forecasting.")
    enable_forecast = False

if enable_forecast:

    # Use full data for forecasting (no 80% split)
    train = series_pos.copy()
    n = len(series_pos)
    split_point = max(3, int(n * 0.8))

    test = series_pos.iloc[split_point:].copy()

    if len(train) < 3 or len(test) == 0:
        st.info("Not enough data for forecasting.")
    else:
        try:
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
            model.fit(train[["ds", "y"]])

            test = test.copy()
            test["ds"] = pd.to_datetime(test["ds"])

            preds_test = model.predict(test[["ds"]])[["ds", "yhat"]]
            preds_test = preds_test.rename(columns={"yhat": "y_pred"})
            preds_test["ds"] = pd.to_datetime(preds_test["ds"])

            merged = pd.merge(test, preds_test, on="ds", how="left")

            if merged["y_pred"].isna().all():
                merged["y_pred"] = model.predict(test[["ds"]])["yhat"].values

            merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce").fillna(1.0)
            merged["y_true"] = merged["y"].clip(lower=1.0)
            merged["y_pred"] = merged["y_pred"].clip(lower=1.0)

            errors = merged["y_true"] - merged["y_pred"]
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            r2 = float(r2_score(merged["y_true"], merged["y_pred"])) if len(merged) > 1 else 0.0

            with np.errstate(divide='ignore', invalid='ignore'):
                mape = float(np.nanmean(np.abs(errors / merged["y_true"])) * 100)
            if np.isnan(mape): mape = 0.0

            colA, colB, colC, colD = st.columns(4)
            colA.metric("MAE", f"{mae:.2f}")
            colB.metric("RMSE", f"{rmse:.2f}")
            colC.metric("R²", f"{r2:.3f}")
            colD.metric("MAPE", f"{mape:.1f}%")

            st.write("Lower MAE/RMSE means better accuracy. Use these numbers for direction, not exact values.")

            st.subheader("Actual vs Predicted")

            show_n = min(12, len(merged))
            view = merged.tail(show_n).copy()

            view["y_true"] = view["y_true"].clip(lower=1)
            view["y_pred"] = view["y_pred"].clip(lower=1)

            labels_cmp = view["ds"].dt.strftime("%Y-%m-%d").tolist()
            x = np.arange(len(labels_cmp))
            width = 0.35

            fig_cmp, ax_cmp = plt.subplots(figsize=(10, 3.8))

            bars_act = ax_cmp.bar(x - width/2, view["y_true"], width,
                                 label="Actual", color="#1565C0", edgecolor="black")
            bars_pred = ax_cmp.bar(x + width/2, view["y_pred"], width,
                                   label="Predicted", color="#FF8F00", edgecolor="black")

            y_max = max(view["y_true"].max(), view["y_pred"].max())
            pad = max(1, 0.02 * y_max)

            for bar in bars_act:
                ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pad,
                            f"{int(bar.get_height()):,}",
                            ha="center", va="bottom", fontsize=9, color="#0D47A1")

            for bar in bars_pred:
                ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pad,
                            f"{int(bar.get_height()):,}",
                            ha="center", va="bottom", fontsize=9, color="#E65100")

            ax_cmp.set_ylabel("Value")
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(labels_cmp, rotation=40, ha="right")
            ax_cmp.set_ylim(0, y_max * 1.3)
            ax_cmp.legend()

            plt.tight_layout()
            st.pyplot(fig_cmp)

            st.subheader("Forecasted Sales (Next Months)")

            future = model.make_future_dataframe(periods=int(forecast_days))
            forecast_df = model.predict(future)[["ds", "yhat"]].rename(columns={"yhat": "y_pred"})

            future_only = forecast_df.tail(int(forecast_days)).copy()
            future_only["y_pred"] = future_only["y_pred"].clip(lower=1.0)

            future_only["month"] = future_only["ds"].dt.to_period("M").astype(str)
            monthly_future = future_only.groupby("month")["y_pred"].mean().reset_index()

            if len(monthly_future) < 3:
                f = future_only.reset_index(drop=True)
                k = min(4, len(f))
                buckets = np.array_split(f, k)
                labels = []
                vals = []
                for b in buckets:
                    start = b["ds"].dt.strftime("%b %d").iloc[0]
                    end = b["ds"].dt.strftime("%b %d").iloc[-1]
                    labels.append(f"{start} → {end}")
                    vals.append(b["y_pred"].mean())
                monthly_future = pd.DataFrame({"month": labels, "y_pred": vals})


            fig_fm, ax_fm = plt.subplots(figsize=(10, 3.8))

            bars = ax_fm.bar(range(len(monthly_future)),
                             monthly_future["y_pred"],
                             color="blue", edgecolor="black")

            y_max2 = monthly_future["y_pred"].max()
            
            pad2=max(1,0.02*y_max2)
            for i, bar in enumerate(bars):
                ax_fm.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + pad2,
                           f"{int(bar.get_height()):,}",
                           ha="center", va="bottom", fontsize=10)

            ax_fm.set_xticks(range(len(monthly_future)))
            ax_fm.set_xticklabels(monthly_future["month"], rotation=40, ha="right")
            ax_fm.set_title("Forecasted Sales (Monthly Summary)")
            ax_fm.set_ylabel("Predicted Value")

            plt.tight_layout()
            st.pyplot(fig_fm)

            csv = monthly_future.to_csv(index=False)
            st.download_button("Download Monthly Forecast CSV", csv,
                               file_name="monthly_forecast.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Forecast failed: {e}")

else:
    st.info("Enable forecasting from the sidebar to generate predictions.")

st.markdown("---")
st.subheader("Smart Insights (AI Powered)")

sample_text = series_pos.head(6).to_string(index=False)

top_items_text = ""
try:
    if selected_cat:
        prod_sums = (
            df_focus.groupby(selected_cat)[value_col]
            .sum()
            .sort_values(ascending=False)
        )
        top_items_text = prod_sums.head(5).to_string()
except Exception:
    top_items_text = ""

if groq_client is None:
    st.info("AI is not configured. Add GROQ_API_KEY to enable AI features.")
else:
    st.markdown(
        "<div class='small-muted'>AI is active — you can use all insights below.</div>",
        unsafe_allow_html=True,
    )

if run_ai_now:

    if groq_client is None:
        st.error("AI is not configured.")
    else:
        st.subheader("AI Insights (Plain English)")

        if ai_explain_forecast:
            st.write("### AI — Forecast Explanation")
            prompt = f"""
Explain the forecast results in simple, everyday English.
Do NOT use technical words.

Include:
• Is the trend going up or down?
• Is the model roughly accurate?
• Why MAE/RMSE might be high or low.

Metrics:
MAE={mae}, RMSE={rmse}, R2={r2}

Recent sample:
{sample_text}
"""
            with st.spinner("AI is explaining forecast..."):
                out = ask_ai(prompt, max_tokens=350)
            st.write(out)

        if ai_exec_summary:
            st.write("### AI — Executive Summary")
            prompt = f"""
Write a simple 3-bullet summary for senior managers.

Include:
• 1 key observation  
• 1 risk  
• 3 easy next steps  

Data sample:
{sample_text}

Top items:
{top_items_text}
"""
            with st.spinner("AI is writing summary..."):
                out = ask_ai(prompt, max_tokens=350)
            st.write(out)

        # 3 — Detect Anomalies
        if ai_detect_anomalies:
            st.write("### AI — Anomaly Detection")
            snippet = (
                series_pos.sort_values("y", ascending=False)
                .head(60)
                .to_string(index=False)
            )
            prompt = f"""
Look at this data and spot unusual behaviour (big jumps, drops, strange values).
Explain in simple language.

Data:
{snippet}
"""
            with st.spinner("AI is detecting anomalies..."):
                out = ask_ai(prompt, max_tokens=350)
            st.write(out)

        if ai_marketing:
            st.write("### AI — Growth Suggestion")
            prompt = f"""
Give one simple, practical suggestion to improve sales or growth.

Data sample:
{sample_text}
"""
            with st.spinner("AI is preparing suggestion..."):
                out = ask_ai(prompt, max_tokens=250)
            st.write(out)

        if ai_compare:
            st.write("### AI — Comparison")
            if selected_cat:
                prompt = f"""
Compare the top items listed below.

Explain:
• Which is performing best?
• One improvement idea for each item.
• Use easy, everyday language.

Top items:
{top_items_text}
"""
            else:
                prompt = "No category selected."

            with st.spinner("AI is comparing items..."):
                out = ask_ai(prompt, max_tokens=350)
            st.write(out)

        if ai_seasonality:
            st.write("### AI — Weekly Pattern Explanation")
            try:
                weekly_vals = weekly.tolist()
            except NameError:
                weekly_vals = "N/A"
                
            prompt = f"""
Explain the weekly pattern from Monday to Sunday in simple language.

Numbers:
{weekly_vals}
"""
            with st.spinner("AI is explaining weekly pattern..."):
                out = ask_ai(prompt, max_tokens=300)
            st.write(out)

if run_custom_ai:
    if not custom_ai_question.strip():
        st.warning("Please type a question first.")
    elif groq_client is None:
        st.error("AI is not configured.")
    else:
        st.subheader("AI — Your Question Answered")
        prompt = f"""
Answer the user’s question in plain English, very simple sentences.

User question:
{custom_ai_question}

Context data:
{sample_text}
"""
        with st.spinner("AI is answering..."):
            out = ask_ai(prompt, max_tokens=500)
        st.write(out)
