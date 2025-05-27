import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NPRS-P Prototype v0.6")
st.markdown("Parse FIT files and classify terrain segments automatically")

uploaded_file = st.file_uploader("📂 Upload FIT file", type=["fit"])

def smooth_series(series, window=15):
    return series.rolling(window=window, min_periods=1, center=True).mean()

def parse_fit_to_df(fit_file):
    records = []

    with fitdecode.FitReader(fit_file) as fit:
        for frame in fit:
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
                record = {field.name: field.value for field in frame.fields}
                records.append(record)

    df = pd.DataFrame(records)

    required_cols = ['timestamp', 'enhanced_altitude', 'power', 'distance']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    df = df[required_cols].dropna()

    df['delta_altitude'] = df['enhanced_altitude'].diff()
    df['delta_distance'] = df['distance'].diff()
    df['gradient'] = df['delta_altitude'] / df['delta_distance'].replace(0, np.nan)

    def classify_segment(g):
        if g > 0.03:
            return 'uphill'
        elif g < -0.03:
            return 'downhill'
        else:
            return 'flat'

    df['segment'] = df['gradient'].apply(classify_segment)

    return df

if uploaded_file is not None:
    with st.spinner('Processing...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Failed to parse the file. Check file and columns.")
        st.stop()

    weight = st.number_input("⚖️ Enter your weight in kg", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    df['w_per_kg'] = df['power'] / weight

    segments = ['uphill', 'flat', 'downhill']
    segment_dfs = {}

    for seg in segments:
        seg_df = df[df['segment'] == seg].copy()
        seg_df = seg_df[seg_df['power'] > 0].reset_index(drop=True)
        if not seg_df.empty:
            seg_df['timestamp'] = pd.to_datetime(seg_df['timestamp'])
            seg_df['elapsed_time_sec'] = (seg_df['timestamp'] - seg_df['timestamp'].iloc[0]).dt.total_seconds()
            seg_df['w_per_kg_smooth'] = smooth_series(seg_df['w_per_kg'], window=15)
            segment_dfs[seg] = seg_df
        else:
            segment_dfs[seg] = pd.DataFrame()

    st.subheader("🧮 Average W/kg by Terrain (NRRS-P)")
    mean_wkg = {seg: segment_dfs[seg]['w_per_kg'].mean() if not segment_dfs[seg].empty else 0 for seg in segments}
    st.write({k: round(v, 2) for k, v in mean_wkg.items()})

    st.subheader("📊 Terrain-wise Power (W/kg) Trend: Scatter & Smoothed Line + Average Line")
    fig, axs = plt.subplots(len(segments), 1, figsize=(12, 8), sharex=True)
    for i, seg in enumerate(segments):
        ax = axs[i]
        seg_df = segment_dfs[seg]
        if not seg_df.empty:
            avg = seg_df['w_per_kg'].mean()
            ax.scatter(seg_df['elapsed_time_sec'], seg_df['w_per_kg'], s=5, alpha=0.4, label='Raw Data')
            ax.plot(seg_df['elapsed_time_sec'], seg_df['w_per_kg_smooth'],  color='blue', label='Smoothed')
            ax.axhline(avg, color='red', linestyle='--', label=f'Average ({avg:.2f})')
            ax.set_ylabel(f"{seg.capitalize()} W/kg")
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No data for {seg}", ha='center', va='center')
            ax.set_ylabel(f"{seg.capitalize()} W/kg")


    
    axs[-1].set_xlabel("Segment Elapsed Time (seconds)")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("📁 Download CSV")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Save as CSV", csv_data, file_name="nprs_parsed.csv", mime="text/csv")
