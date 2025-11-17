import os
from datetime import datetime, date as dt_date, time as dt_time, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from supabase import create_client, Client

# ------------------------
# streamlit + supabase setup
# ------------------------

st.set_page_config(page_title="metabolic coach (supabase)", layout="wide")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("SUPABASE_URL and SUPABASE_ANON_KEY must be set as environment variables.")
    st.stop()


@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


supabase = get_supabase()

# ------------------------
# ui header
# ------------------------

st.title("metabolic coach v2 (supabase)")

st.markdown(
    """
this app stores everything in supabase:

- glucose readings from your LibreView CSV (mmol/L)
- events (meals, snacks, Meiji Meibalance, drinks)

you can:
- upload glucose data once per sensor cycle
- log food and drinks from any device
- see the same data on laptop and phone
"""
)

# ------------------------
# sidebar settings
# ------------------------

st.sidebar.header("settings")

user_id = st.sidebar.text_input("user id", value="zach").strip()
if not user_id:
    st.sidebar.error("user id cannot be empty.")
    st.stop()

unit = st.sidebar.selectbox("display glucose unit", ["mmol/L", "mg/dL"])

# note: source data is mmol/L, but you can view in mg/dL later if you want
if unit == "mg/dL":
    target_low = st.sidebar.number_input("target low", value=70.0)
    target_high = st.sidebar.number_input("target high", value=180.0)
else:
    target_low = st.sidebar.number_input("target low", value=3.9)
    target_high = st.sidebar.number_input("target high", value=10.0)

event_window_hours = st.sidebar.slider(
    "event impact window (hours)", 1.0, 4.0, 2.0, 0.5
)
st.sidebar.caption("shorter window for drinks, longer for big meals")

st.sidebar.divider()
st.sidebar.write(f"current user: **{user_id}**")


# ------------------------
# supabase helpers
# ------------------------

def parse_libre_csv_to_df(file) -> pd.DataFrame:
    """
    parse LibreView CSV (mmol/L) into a long dataframe with:
    timestamp, glucose, source, date, hour
    expects columns:
    - Device Timestamp
    - Historic Glucose mmol/L
    - Scan Glucose mmol/L
    - Strip Glucose mmol/L (optional)
    """
    df = pd.read_csv(file, engine="python")

    time_col = "Device Timestamp"
    glucose_cols = [
        "Historic Glucose mmol/L",
        "Scan Glucose mmol/L",
        "Strip Glucose mmol/L",
    ]
    available_glucose_cols = [c for c in glucose_cols if c in df.columns]

    if time_col not in df.columns or not available_glucose_cols:
        raise ValueError(
            "could not find 'Device Timestamp' or any mmol/L glucose columns "
            "(Historic / Scan / Strip) in your LibreView CSV."
        )

    df_melt = df.melt(
        id_vars=[time_col],
        value_vars=available_glucose_cols,
        var_name="source",
        value_name="glucose",
    )

    df_melt = df_melt.dropna(subset=["glucose"])
    df_melt["timestamp"] = pd.to_datetime(
        df_melt[time_col], errors="coerce", dayfirst=True
    )
    df_melt = df_melt.dropna(subset=["timestamp"])

    df_melt["glucose"] = pd.to_numeric(df_melt["glucose"], errors="coerce")
    df_melt = df_melt.dropna(subset=["glucose"])

    df_melt = df_melt.sort_values("timestamp").reset_index(drop=True)
    df_melt["date"] = df_melt["timestamp"].dt.date
    df_melt["hour"] = (
        df_melt["timestamp"].dt.hour + df_melt["timestamp"].dt.minute / 60.0
    )

    return df_melt[["timestamp", "glucose", "source", "date", "hour"]]


def upsert_glucose_data(user_id: str, glucose_df: pd.DataFrame):
    """
    clear old glucose rows for user_id and insert new ones
    into glucose_readings table.
    expected schema:
    - user_id text
    - timestamp timestamptz
    - glucose double precision
    - source text
    """
    if glucose_df.empty:
        return

    supabase.table("glucose_readings").delete().eq("user_id", user_id).execute()

    records = []
    for _, row in glucose_df.iterrows():
        records.append(
            {
                "user_id": user_id,
                "timestamp": row["timestamp"].isoformat(),
                "glucose": float(row["glucose"]),
                "source": str(row["source"]),
            }
        )

    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        supabase.table("glucose_readings").insert(chunk).execute()


def fetch_glucose_df(user_id: str) -> pd.DataFrame:
    res = (
        supabase.table("glucose_readings")
        .select("timestamp, glucose, source")
        .eq("user_id", user_id)
        .order("timestamp", desc=False)
        .execute()
    )
    data = res.data or []
    if not data:
        return pd.DataFrame(columns=["timestamp", "glucose", "source", "date", "hour"])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df = df.dropna(subset=["glucose"])

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "glucose", "source", "date", "hour"]]


def fetch_events_df(user_id: str) -> pd.DataFrame:
    res = (
        supabase.table("events")
        .select("timestamp, label, tags")
        .eq("user_id", user_id)
        .order("timestamp", desc=False)
        .execute()
    )
    data = res.data or []
    if not data:
        return pd.DataFrame(columns=["timestamp", "label", "tags"])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "tags" not in df.columns:
        df["tags"] = ""
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df[["timestamp", "label", "tags"]]


def log_event(user_id: str, ts: datetime, label: str, tags: str):
    supabase.table("events").insert(
        {
            "user_id": user_id,
            "timestamp": ts.isoformat(),
            "label": label,
            "tags": tags,
        }
    ).execute()


def update_event(
    user_id: str,
    original_ts: datetime,
    original_label: str,
    new_ts: datetime,
    new_label: str,
    new_tags: str,
):
    supabase.table("events").update(
        {
            "timestamp": new_ts.isoformat(),
            "label": new_label,
            "tags": new_tags,
        }
    ).eq("user_id", user_id).eq("timestamp", original_ts.isoformat()).eq(
        "label", original_label
    ).execute()


def delete_event(user_id: str, original_ts: datetime, original_label: str):
    supabase.table("events").delete().eq("user_id", user_id).eq(
        "timestamp", original_ts.isoformat()
    ).eq("label", original_label).execute()


# ------------------------
# analysis helpers
# ------------------------

def compute_cgm_metrics(df: pd.DataFrame, target_low: float, target_high: float):
    total_points = len(df)
    if total_points == 0:
        return {}

    in_range = df[(df["glucose"] >= target_low) & (df["glucose"] <= target_high)]
    low = df[df["glucose"] < target_low]
    high = df[df["glucose"] > target_high]

    overall = {
        "total_points": total_points,
        "time_in_range_pct": len(in_range) / total_points * 100 if total_points else 0,
        "low_pct": len(low) / total_points * 100 if total_points else 0,
        "high_pct": len(high) / total_points * 100 if total_points else 0,
        "mean_glucose": df["glucose"].mean() if total_points else 0,
        "std_glucose": df["glucose"].std() if total_points else 0,
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


def compute_event_impacts(
    cgm_df: pd.DataFrame, events_df: pd.DataFrame, window_hours: float
):
    if events_df is None or events_df.empty or cgm_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    window = timedelta(hours=window_hours)
    pre_window = timedelta(minutes=30)

    for _, ev in events_df.iterrows():
        t = ev["timestamp"]
        label = ev["label"]
        tags = ev.get("tags", "")

        pre_period = cgm_df[
            (cgm_df["timestamp"] >= t - pre_window)
            & (cgm_df["timestamp"] < t)
        ]
        post_period = cgm_df[
            (cgm_df["timestamp"] >= t) & (cgm_df["timestamp"] <= t + window)
        ]

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


def compute_time_of_day_sensitivity(cgm_df: pd.DataFrame):
    if cgm_df is None or cgm_df.empty:
        return pd.DataFrame()
    df = cgm_df.copy()
    overall_mean = df["glucose"].mean()
    df["delta_from_mean"] = df["glucose"] - overall_mean
    df["hour"] = df["timestamp"].dt.hour
    by_hour = df.groupby("hour")["delta_from_mean"].mean().reset_index()
    by_hour.rename(columns={"delta_from_mean": "avg_delta_from_mean"}, inplace=True)
    return by_hour


def generate_coaching_summary(
    cgm_metrics, event_summary, unit: str, target_low: float, target_high: float
):
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
        parts.append(
            f"about {low_pct:.1f} percent of readings are below target, so lows are something to watch."
        )
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
            f"your best day was {best_row['date']} with a day score of {best_row['day_score']:.0f} "
            f"and time in range {best_row['time_in_range_pct']:.1f} percent."
        )
        parts.append(
            f"the most challenging day was {worst_row['date']} with a day score of {worst_row['day_score']:.0f}."
        )

    if event_summary is not None and not event_summary.empty:
        meiji_rows = event_summary[
            event_summary["label"].str.contains("Meiji Meibalance", case=False)
        ]
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


# ------------------------
# step 1. upload Libre CSV to sync glucose
# ------------------------

st.subheader("step 1. upload LibreView CSV (mmol/L) to sync glucose into supabase")

libre_file = st.file_uploader("LibreView CSV", type=["csv"], key="glucose_csv")

if libre_file is not None:
    try:
        parsed_df = parse_libre_csv_to_df(libre_file)
        upsert_glucose_data(user_id, parsed_df)
        st.success(
            f"synced {len(parsed_df)} glucose readings to supabase for user '{user_id}'."
        )
    except Exception as e:
        st.error(f"error parsing or syncing LibreView CSV: {e}")

st.markdown(
    "if you already uploaded glucose data for this user id before, you can skip this step and just scroll down."
)
st.divider()

# ------------------------
# step 2. log events
# ------------------------

st.subheader("step 2. log food and drinks")

events_df = fetch_events_df(user_id)

with st.form("log_event_form"):
    now = datetime.now()
    col_date, col_time = st.columns(2)
    date_val = col_date.date_input("date", value=now.date())
    time_val = col_time.time_input(
        "time", value=now.time().replace(second=0, microsecond=0)
    )

    preset = st.selectbox(
        "quick label",
        ["Meiji Meibalance", "meal", "snack", "drink", "custom"],
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

        log_event(user_id, ts, label_val, tags_val)
        st.success(f"event logged: {label_val} at {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        st.rerun()

st.markdown("recent events")
if events_df.empty:
    st.info("no events logged yet.")
else:
    st.dataframe(events_df[["timestamp", "label", "tags"]].head(30))

    st.subheader("edit or delete an event")

    ev_with_idx = events_df.reset_index()  # adds 'index' column
    options = [
        f"{row['index']}: {row['timestamp'].strftime('%Y-%m-%d %H:%M')} - {row['label']}"
        for _, row in ev_with_idx.iterrows()
    ]

    selected = st.selectbox("choose event to edit", options) if options else None

    if selected:
        idx_str = selected.split(":", 1)[0]
        try:
            idx = int(idx_str)
        except ValueError:
            idx = 0

        if 0 <= idx < len(events_df):
            ev = events_df.iloc[idx]

            col_edate, col_etime = st.columns(2)
            edit_date = col_edate.date_input(
                "edit date", value=ev["timestamp"].date(), key=f"edit_date_{idx}"
            )
            edit_time = col_etime.time_input(
                "edit time", value=ev["timestamp"].time(), key=f"edit_time_{idx}"
            )

            edit_label = st.text_input(
                "edit label", value=ev["label"], key=f"edit_label_{idx}"
            )
            edit_tags = st.text_input(
                "edit tags", value=ev["tags"], key=f"edit_tags_{idx}"
            )

            c1, c2 = st.columns(2)
            if c1.button("save changes"):
                new_ts = datetime.combine(edit_date, edit_time)
                update_event(
                    user_id,
                    original_ts=ev["timestamp"],
                    original_label=ev["label"],
                    new_ts=new_ts,
                    new_label=edit_label.strip(),
                    new_tags=edit_tags.strip(),
                )
                st.success("event updated.")
                st.rerun()

            if c2.button("delete event"):
                delete_event(
                    user_id,
                    original_ts=ev["timestamp"],
                    original_label=ev["label"],
                )
                st.warning("event deleted.")
                st.rerun()

st.divider()

# ------------------------
# step 3. glucose analysis from supabase
# ------------------------

glucose_df = fetch_glucose_df(user_id)

if glucose_df.empty:
    st.warning(
        "no glucose data found for this user. upload a LibreView CSV above to sync."
    )
    st.stop()

cgm_metrics = compute_cgm_metrics(glucose_df, target_low, target_high)
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
unique_dates = sorted(glucose_df["date"].unique())
selected_date = st.selectbox(
    "select a date", unique_dates, index=len(unique_dates) - 1
)
day_data = glucose_df[glucose_df["date"] == selected_date]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(day_data["timestamp"], day_data["glucose"])
ax.axhline(target_low, linestyle="--", label="target low")
ax.axhline(target_high, linestyle="--", label="target high")
ax.set_title(f"glucose on {selected_date}")
ax.set_xlabel("time")
ax.set_ylabel(f"glucose ({unit})")
ax.legend()
st.pyplot(fig)

events_metrics, event_summary = compute_event_impacts(
    glucose_df, events_df, event_window_hours
)

if events_metrics is None or events_metrics.empty:
    st.info(
        "no event impacts computed yet. log some events and make sure there is CGM data around them."
    )
else:
    st.subheader("event impacts")
    st.markdown("per event instance")
    st.dataframe(events_metrics.sort_values("timestamp"))

    st.markdown("average impact per label")
    st.dataframe(event_summary.sort_values("avg_delta_peak", ascending=False))

    meiji_ev = events_metrics[
        events_metrics["label"].str.contains("Meiji Meibalance", case=False)
    ]
    if not meiji_ev.empty:
        st.subheader("Meiji Meibalance response examples")
        st.dataframe(meiji_ev.sort_values("timestamp"))

st.subheader("time of day glucose sensitivity (rough)")
tod_df = compute_time_of_day_sensitivity(glucose_df)
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
summary_text = generate_coaching_summary(
    cgm_metrics, event_summary, unit, target_low, target_high
)
st.write(summary_text)

st.caption("metabolic coach v2 (supabase). glucose + events stored per user id.")