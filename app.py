import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P Prototype v0.6")
st.markdown("One-shot analysis of FIT files with terrain classification")

uploaded_file = st.file_uploader("📂 Upload your FIT file", type=["fit"])

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

    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df = df.dropna(subset=['power'])
    df = df[df['power'] > 0]
    df = df.reset_index(drop=True)
    
    return df

if uploaded_file is not None:
    st.sidebar.subheader("Input your weight (kg)")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    with st.spinner('Analyzing...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Failed to parse the file or missing required columns.")
        st.stop()

    # Remove zero or less power values
    df = df[df['power'] > 0].copy()

    df['w_per_kg'] = df['power'] / weight

    st.success("✅ Analysis completed!")

    st.subheader("=== Columns ===")
    st.write(df.columns.tolist())

    st.subheader("📊 Data preview (first 100 rows)")
    st.dataframe(df.head(100))

    # Terrain-wise average W/kg (NRRS-P)
    st.subheader("🧮 Average W/kg by Terrain (NRRS-P)")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    # Prepare dataframes by terrain
    df_uphill = df[df['segment'] == 'uphill']
    df_flat = df[df['segment'] == 'flat']
    df_downhill = df[df['segment'] == 'downhill']

    # Create a function for smoothed line (rolling mean)
    def smooth(series, window=10):
        return series.rolling(window=window, min_periods=1).mean()

    # Plot integrated graph
    st.subheader("📈 Integrated W/kg Scatter & Smoothed Line (All Terrain)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df.index, df['w_per_kg'], s=5, alpha=0.3, label='Raw Data')
    ax.plot(smooth(df['w_per_kg']), color='red', linewidth=2, label='Smoothed')
    ax.set_xlabel("Index")
    ax.set_ylabel("W/kg")
    ax.legend()
    st.pyplot(fig)

    # Plot by terrain
    st.subheader("📈 W/kg Scatter & Smoothed Line by Terrain")

    terrains = {
        'uphill': df_uphill,
        'flat': df_flat,
        'downhill': df_downhill,
        'all': df
    }
    colors = {
        'uphill': 'orange',
        'flat': 'green',
        'downhill': 'blue',
        'all': 'red'
    }

    for terrain, data in terrains.items():
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(data.index, data['w_per_kg'], s=10, alpha=0.3, label='Raw Data')
        ax.plot(smooth(data['w_per_kg']), color=colors[terrain], linewidth=2, label='Smoothed')
        ax.set_title(f"W/kg - {terrain.capitalize()}")
        ax.set_xlabel("Index")
        ax.set_ylabel("W/kg")
        ax.legend()
        st.pyplot(fig)

    # CSV download
    st.subheader("📁 Export CSV")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data, file_name="nrrs_parsed.csv", mime="text/csv")
