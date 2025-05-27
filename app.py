import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P Prototype v5.0")
st.markdown("FIT file analysis with terrain segmentation and smoothed power per kg visualization")

uploaded_file = st.file_uploader("📂 Upload FIT file", type=["fit"])

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

def smooth_series(series, window=10):
    return series.rolling(window, min_periods=1, center=True).mean()

if uploaded_file is not None:
    with st.spinner('Analyzing...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Analysis failed. Check file and columns.")
        st.stop()

    # Input weight
    weight = st.number_input("Enter your weight (kg):", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
    df['w_per_kg'] = df['power'] / weight

    st.success("✅ Analysis Complete!")

    st.subheader("Columns:")
    st.write(df.columns.tolist())

    st.subheader("Raw Data (First 100 rows):")
    st.dataframe(df.head(100))

    st.subheader("Average Power per kg by Terrain Segment (NRRS-P):")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    st.subheader("Power per kg Visualization by Terrain Segment")

    segments = ['uphill', 'flat', 'downhill']

    for seg in segments:
        seg_df = df[df['segment'] == seg].copy()
        seg_df = seg_df[seg_df['power'] > 0]  # filter out power=0 to avoid noise

        if seg_df.empty:
            st.write(f"No data for segment: {seg}")
            continue

        # Create cumulative time axis in seconds (approximate, based on timestamps)
        seg_df['timestamp'] = pd.to_datetime(seg_df['timestamp'])
        seg_df['time_sec'] = (seg_df['timestamp'] - seg_df['timestamp'].iloc[0]).dt.total_seconds()

        # Smoothing W/kg
        seg_df['w_per_kg_smooth'] = smooth_series(seg_df['w_per_kg'], window=15)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.scatter(seg_df['time_sec'], seg_df['w_per_kg'], color='lightblue', alpha=0.5, label='Raw Data')
        ax.plot(seg_df['time_sec'], seg_df['w_per_kg_smooth'], color='red', label='Smoothed Trend', linewidth=2)
        ax.set_title(f"Power per kg on {seg.capitalize()}")
        ax.set_xlabel("Elapsed Time (sec)")
        ax.set_ylabel("Power per kg (W/kg)")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    # CSV download
    st.subheader("Download CSV")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data, file_name="nrrs_p_parsed_v5.csv", mime="text/csv")
