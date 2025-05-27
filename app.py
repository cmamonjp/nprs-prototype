import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P Prototype v0.5")
st.markdown("Upload a FIT file to analyze power per kg with terrain segmentation and smoothing")

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

if uploaded_file is not None:
    st.sidebar.subheader("Input your weight (kg)")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    with st.spinner('Analyzing...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Failed to parse the file or missing required columns.")
        st.stop()

    df['w_per_kg'] = df['power'] / weight

    # Calculate elapsed time in seconds from timestamp (assuming timestamp is datetime64)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['elapsed_time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    window_size = 10

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Smooth the W/kg data for all terrain combined
    df['w_per_kg_smooth'] = df['w_per_kg'].rolling(window=window_size, min_periods=1).mean()

    # 1. All terrain combined
    axs[0].scatter(df['elapsed_time_sec'], df['w_per_kg'], alpha=0.3, s=10, label='Raw Data')
    axs[0].plot(df['elapsed_time_sec'], df['w_per_kg_smooth'], color='blue', linewidth=2, label='Smoothed')
    axs[0].set_title('All Terrain')
    axs[0].set_ylabel('Power per kg (W/kg)')
    axs[0].legend()
    axs[0].grid(True)

    # 2~4. By terrain segments
    for i, terrain in enumerate(['uphill', 'flat', 'downhill'], start=1):
        terrain_df = df[df['segment'] == terrain].copy()
        terrain_df['w_per_kg_smooth'] = terrain_df['w_per_kg'].rolling(window=window_size, min_periods=1).mean()

        axs[i].scatter(terrain_df['elapsed_time_sec'], terrain_df['w_per_kg'], alpha=0.3, s=10, label='Raw Data')
        axs[i].plot(terrain_df['elapsed_time_sec'], terrain_df['w_per_kg_smooth'], color='red', linewidth=2, label='Smoothed')
        axs[i].set_title(f'{terrain.capitalize()} Segment')
        axs[i].set_ylabel('Power per kg (W/kg)')
        axs[i].legend()
        axs[i].grid(True)

    axs[3].set_xlabel('Elapsed Time (sec)')

    st.pyplot(fig)

    # Display average W/kg by terrain
    st.subheader("🧮 Average Power per kg by Terrain Segment")
    avg_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(avg_wkg)

    # Provide CSV download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data, file_name="nrrs_p_output.csv", mime="text/csv")
