import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import os

st.set_page_config(page_title="Metabolic Coach", layout="wide")

EVENTS_PATH = "events.csv"

st.title("metabolic coach v1.1")

st.markdown(
    """
this app uses:

1. your LibreView CSV export (14 days CGM data)
2. a local `events.csv` file that this app manages for you

you can:

- log Meiji Meibalance or any food directly here
- see how each event affects your glucose
- view day scores and a simple coaching summary
"""
)

st.sidebar.header("settings")

unit = st.sidebar.selectbox("glucose unit", ["mg/dL", "mmol/L"])

if unit == "mg/dL":
    target_low = st.sidebar.number_input("target low", value=70)
    target_high = st.sidebar.number_input("target high", value=180)
else:
    target_low = st.sidebar.number_input("target low", value=3.9)
    target_high = st.sidebar.number_input("target high", value=10.0)

event_window_hours = st.sidebar.slider(
    "event impact window (hours)", 1.0, 4.0, 2.0, 0.5
)

st.sidebar.caption("shorter window for drinks, longer for big meals")


libre_file = st.file_uploader("upload LibreView CSV", type=["csv"])


def load_existing_events(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
        else:
            df["timestamp"] = pd.NaT
        if "label" not in df.columns:
            df["label"] = ""
        if "tags" not in df.columns:
            df["tags"] = ""
    else:
        df = pd.DataFrame(columns=["timestamp", "label", "tags"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_events(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


@st.cache_data
def load_libre_data(file):
    df = pd.read_csv(file, engine="python")

    # find columns
    time_col = "Device Timestamp"
    glucose_cols = [
        "Historic Glucose mmol/L",
        "Scan Glucose mmol/L",
        "Strip Glucose mmol/L"
    ]

    # keep only existing glucose columns
    available_glucose_cols = [c for c in glucose_cols if c in df.columns]

    if time_col not in df.columns or not available_glucose_cols:
        raise ValueError("Could not find expected timestamp or glucose columns in your LibreView CSV.")

    # melt the glucose columns into long form
    df_melt = df.melt(
        id_vars=[time_col],
        value_vars=available_glucose_cols,
        var_name="source",
        value_name="glucose"
    )

    # drop rows with no glucose
    df_melt = df_melt.dropna(subset=["glucose"])

    # parse timestamp
    df_melt["timestamp"] = pd.to_datetime(df_melt[time_col], errors="coerce", dayfirst=True)
    df_melt = df_melt.dropna(subset=["timestamp"])

    # convert glucose to float
    df_melt["glucose"] = pd.to_numeric(df_melt["glucose"], errors="coerce")
    df_melt = df_melt.dropna(subset=["glucose"])

    # clean up
    df_melt = df_melt.sort_values("timestamp").reset_index(drop=True)
    df_melt["date"] = df_melt["timestamp"].dt.date
    df_melt["hour"] = df_melt["timestamp"].dt.hour + df_melt["timestamp"].dt.minute / 60.0

    return df_melt



def compute_cgm_metrics(df, target_low, target_high):
    total_points = len(df)
    if total_points == 0:
        return {}

    in_range = df[(df["glucose"] >= target_low) & (df["glucose"] <= target_high)]
    low = df[df["glucose"] < target_low]
    high = df[df["glucose"] > target_high]

    overall = {
        "total_points": total_points,
        "time_in_range_pct": len(in_range) / total_points * 100,
        "low_pct": len(low) / total_points * 100,
        "high_pct": len(high) / total_points * 100,
        "mean_glucose": df["glucose"].mean(),
        "std_glucose": df["glucose"].std(),
    }

    daily = df.groupby("date").agg(
        mean_glucose=("glucose", "mean"),
        max_glucose=("glucose", "max"),
        min_glucose=("glucose", "min"),
        count=("glucose", "count"),
    ).reset_index()

    tir_list = []
    for d, group in df.groupby("date"):
        total = len(group)
        tir = group[(group["glucose"] >= target_low) & (group["glucose"] <= target_high)]
        tir_pct = len(tir) / total * 100 if total > 0 else 0
        tir_list.append({"date": d, "time_in_range_pct": tir_pct})

    tir_df = pd.DataFrame(tir_list)
    daily = daily.merge(tir_df, on="date", how="left")

    scores = []
    for _, row in daily.iterrows():
        tir_score = row["time_in_range_pct"]
        day_data = df[df["date"] == row["date"]]
        total = len(day_data)
        highs = day_data[day_data["glucose"] > target_high]
        high_pct = len(highs) / total * 100 if total > 0 else 0
        penalty = max(0.0, high_pct - 30.0) * 0.5
        score = max(0.0, min(100.0, tir_score - penalty))
        scores.append(score)

    daily["day_score"] = scores

    return {"overall": overall, "daily": daily}


def compute_event_impacts(cgm_df, events_df, window_hours):
    if events_df is None or events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    window = timedelta(hours=window_hours)
    pre_window = timedelta(minutes=30)

    for _, ev in events_df.iterrows():
        t = ev["timestamp"]
        label = ev["label"]
        tags = ev.get("tags", "")

        pre_period = cgm_df[(cgm_df["timestamp"] >= t - pre_window) & (cgm_df["timestamp"] < t)]
        post_period = cgm_df[(cgm_df["timestamp"] >= t) & (cgm_df["timestamp"] <= t + window)]

        if pre_period.empty or post_period.empty:
            continue

        baseline = pre_period["glucose"].mean()
        peak = post_period["glucose"].max()
        nadir = post_period["glucose"].min()
        delta_peak = peak - baseline
        auc = ((post_period["glucose"] - baseline).clip(lower=0)).sum()

        peak_row = post_period[post_period["glucose"] == peak].iloc[0]
        time_to_peak = (peak_row["timestamp"] - t).total_seconds() / 60.0

        results.append(
            {
                "timestamp": t,
                "label": label,
                "tags": tags,
                "baseline": baseline,
                "peak": peak,
                "delta_peak": delta_peak,
                "nadir": nadir,
                "auc_above_baseline": auc,
                "time_to_peak_min": time_to_peak,
            }
        )

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    events_metrics = pd.DataFrame(results)

    summary = (
        events_metrics.groupby("label")
        .agg(
            n=("delta_peak", "count"),
            avg_delta_peak=("delta_peak", "mean"),
            avg_peak=("peak", "mean"),
            avg_baseline=("baseline", "mean"),
            avg_auc=("auc_above_baseline", "mean"),
            avg_time_to_peak_min=("time_to_peak_min", "mean"),
        )
        .reset_index()
    )

    return events_metrics, summary


def compute_time_of_day_sensitivity(cgm_df):
    if cgm_df is None or cgm_df.empty:
        return pd.DataFrame()
    df = cgm_df.copy()
    overall_mean = df["glucose"].mean()
    df["delta_from_mean"] = df["glucose"] - overall_mean
    df["hour"] = df["timestamp"].dt.hour
    by_hour = df.groupby("hour")["delta_from_mean"].mean().reset_index()
    by_hour.rename(columns={"delta_from_mean": "avg_delta_from_mean"}, inplace=True)
    return by_hour


def generate_coaching_summary(cgm_metrics, event_summary, unit, target_low, target_high):
    if not cgm_metrics:
        return "no data to summarise."

    o = cgm_metrics["overall"]
    tir = o["time_in_range_pct"]
    low_pct = o["low_pct"]
    high_pct = o["high_pct"]
    mean_g = o["mean_glucose"]

    parts = []
    parts.append(
        f"over this period, your time in range between {target_low} and {target_high} {unit} is about {tir:.1f} percent."
    )
    if low_pct > 5:
        parts.append(f"about {low_pct:.1f} percent of readings are below target, so lows are something to watch.")
    if high_pct > 25:
        parts.append(
            f"around {high_pct:.1f} percent of readings are above target, which suggests frequent spikes that you may want to flatten."
        )
    parts.append(f"average glucose sits around {mean_g:.1f} {unit}.")

    daily = cgm_metrics["daily"]
    if not daily.empty:
        best_row = daily.sort_values("day_score", ascending=False).iloc[0]
        worst_row = daily.sort_values("day_score", ascending=True).iloc[0]
        parts.append(
            f"your best day was {best_row['date']} with a day score of {best_row['day_score']:.0f} and time in range {best_row['time_in_range_pct']:.1f} percent."
        )
        parts.append(
            f"the most challenging day was {worst_row['date']} with a day score of {worst_row['day_score']:.0f}."
        )

    if event_summary is not None and not event_summary.empty:
        meiji_rows = event_summary[event_summary["label"].str.contains("Meiji Meibalance", case=False)]
        if not meiji_rows.empty:
            row = meiji_rows.iloc[0]
            parts.append(
                f"for Meiji Meibalance, the average spike above baseline is about {row['avg_delta_peak']:.1f} {unit}, "
                f"with peak arriving roughly {row['avg_time_to_peak_min']:.0f} minutes after drinking."
            )
            if unit == "mg/dL" and row["avg_delta_peak"] > 40:
                parts.append(
                    "that is a fairly strong spike, so you may want to drink it with food or add a short walk afterwards."
                )
            elif unit == "mmol/L" and row["avg_delta_peak"] > 2:
                parts.append(
                    "that is a fairly strong spike, so you may want to drink it with food or add a short walk afterwards."
                )
            else:
                parts.append(
                    "the spike from Meibalance looks moderate, which is relatively friendly for a supplemental drink."
                )

    return " ".join(parts)


if libre_file is None:
    st.info("upload your LibreView CSV to begin.")
    st.stop()

try:
    libre_df = load_libre_data(libre_file)
except Exception as e:
    st.error(f"error loading LibreView CSV: {e}")
    st.stop()

events_df = load_existing_events(EVENTS_PATH)

# logging ui
st.subheader("log food and drinks")

with st.form("log_event_form"):
    now = datetime.now()
    d_col, t_col = st.columns(2)
    date_val = d_col.date_input("date", value=now.date())
    time_val = t_col.time_input("time", value=now.time().replace(second=0, microsecond=0))

    preset = st.selectbox(
        "quick label",
        [
            "Meiji Meibalance",
            "meal",
            "snack",
            "drink",
            "custom",
        ],
    )
    custom_label = st.text_input("label detail (eg. ramen, rice, etc)", "")

    tags_val = st.text_input("tags (comma separated, optional)", "")

    submitted = st.form_submit_button("add event")

    if submitted:
        ts = datetime.combine(date_val, time_val)
        if preset == "custom" and custom_label.strip():
            label_val = custom_label.strip()
        elif preset != "custom" and custom_label.strip():
            label_val = f"{preset}: {custom_label.strip()}"
        else:
            label_val = preset

        new_row = {"timestamp": ts, "label": label_val, "tags": tags_val}
        events_df = pd.concat([events_df, pd.DataFrame([new_row])], ignore_index=True)
        events_df = events_df.sort_values("timestamp").reset_index(drop=True)
        save_events(events_df, EVENTS_PATH)
        st.success(f"event logged: {label_val} at {ts}")

st.markdown("recent events")
if events_df.empty:
    st.info("no events logged yet.")
else:
    st.dataframe(events_df.sort_values("timestamp", ascending=False).head(20))

# cgm metrics
cgm_metrics = compute_cgm_metrics(libre_df, target_low, target_high)
if not cgm_metrics:
    st.warning("no valid CGM data found.")
    st.stop()

overall = cgm_metrics["overall"]
daily = cgm_metrics["daily"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("time in range", f"{overall['time_in_range_pct']:.1f} %")
c2.metric("low", f"{overall['low_pct']:.1f} %")
c3.metric("high", f"{overall['high_pct']:.1f} %")
c4.metric("mean glucose", f"{overall['mean_glucose']:.1f} {unit}")

st.subheader("daily scores")
st.dataframe(daily.sort_values("date", ascending=False))

st.subheader("daily glucose chart")
unique_dates = sorted(libre_df["date"].unique())
selected_date = st.selectbox("select a date", unique_dates, index=len(unique_dates) - 1)
day_data = libre_df[libre_df["date"] == selected_date]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(day_data["timestamp"], day_data["glucose"])
ax.axhline(target_low, linestyle="--", label="target low")
ax.axhline(target_high, linestyle="--", label="target high")
ax.set_title(f"glucose on {selected_date}")
ax.set_xlabel("time")
ax.set_ylabel(f"glucose ({unit})")
ax.legend()
st.pyplot(fig)

events_metrics, event_summary = compute_event_impacts(libre_df, events_df, event_window_hours)

if events_metrics is None or events_metrics.empty:
    st.info("no event impacts computed yet. log some events and make sure there is CGM data around them.")
else:
    st.subheader("event impacts")
    st.markdown("per event instance")
    st.dataframe(events_metrics.sort_values("timestamp"))

    st.markdown("average impact per label")
    st.dataframe(event_summary.sort_values("avg_delta_peak", ascending=False))

    meiji_ev = events_metrics[events_metrics["label"].str.contains("Meiji Meibalance", case=False)]
    if not meiji_ev.empty:
        st.subheader("Meiji Meibalance response examples")
        st.dataframe(meiji_ev.sort_values("timestamp"))

st.subheader("time of day glucose sensitivity (rough)")
tod_df = compute_time_of_day_sensitivity(libre_df)
if tod_df is not None and not tod_df.empty:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(tod_df["hour"], tod_df["avg_delta_from_mean"])
    ax2.set_xlabel("hour of day")
    ax2.set_ylabel("avg deviation from mean")
    ax2.set_title("average glucose deviation by hour")
    st.pyplot(fig2)
else:
    st.info("not enough data to compute time of day pattern.")

st.subheader("coaching summary")
summary_text = generate_coaching_summary(cgm_metrics, event_summary, unit, target_low, target_high)
st.write(summary_text)

st.caption("metabolic coach v1.1. events are saved to events.csv in this folder.")
