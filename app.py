import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P Prototype v0.4")
st.markdown("From FIT file to terrain segmentation and power evaluation")

uploaded_file = st.file_uploader("📂 Upload your FIT file", type=["fit"])

# ----------------------------
# Parse FIT file to DataFrame
# ----------------------------
def parse_fit_to_df(fit_file):
    records = []

    with fitdecode.FitReader(fit_file) as fit:
        for frame in fit:
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
                record = {field.name: field.value for field in frame.fields}
                records.append(record)

    df = pd.DataFrame(records)

    # Required columns
    required_cols = ['timestamp', 'enhanced_altitude', 'power', 'distance']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return pd.DataFrame()

    df = df[required_cols].dropna()

    # Calculate gradient
    df['delta_altitude'] = df['enhanced_altitude'].diff()
    df['delta_distance'] = df['distance'].diff()
    df['gradient'] = df['delta_altitude'] / df['delta_distance'].replace(0, np.nan)

    def classify(g):
        if g > 0.03:
            return 'uphill'
        elif g < -0.03:
            return 'downhill'
        else:
            return 'flat'

    df['segment'] = df['gradient'].apply(classify)

    return df

# ----------------------------
# Main process
# ----------------------------
if uploaded_file is not None:
    with st.spinner('Parsing FIT file...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Parsing failed. Please check the input file.")
        st.stop()

    st.success("✅ Data loaded successfully!")

    # Body weight input
    st.subheader("⚖️ Enter your weight (kg)")
    weight = st.number_input("Body weight", min_value=30.0, max_value=120.0, value=60.0, step=0.5)

    # Add W/kg column
    df['w_per_kg'] = df['power'] / weight

    # Filter out power == 0
    df = df[df['power'] > 0]

    if df.empty:
        st.warning("Only power=0 data found. Nothing to analyze.")
        st.stop()

    # Column info
    st.subheader("📋 Column names")
    st.write(df.columns.tolist())

    # Show sample data
    st.subheader("📊 Head of DataFrame (top 100)")
    st.dataframe(df.head(100))

    # W/kg mean by terrain
    st.subheader("🧮 Mean W/kg by terrain type")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    # Scatter plot by segment
    st.subheader("📈 Scatter plot of Power by segment")
    for seg_type in ['uphill', 'flat', 'downhill']:
        seg = df[df['segment'] == seg_type].copy()
        if seg.empty:
            continue
        seg['elapsed_time'] = (seg['timestamp'] - seg['timestamp'].iloc[0]).dt.total_seconds()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(seg['elapsed_time'], seg['power'], s=3, alpha=0.6, label=seg_type, c='tab:blue')
        ax.set_title(f"{seg_type.capitalize()} segment")
        ax.set_xlabel("Elapsed time (sec)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        st.pyplot(fig)

    # CSV download
    st.subheader("📁 Export cleaned data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download CSV", csv_data, file_name="nrrs_p_cleaned.csv", mime="text/csv")
