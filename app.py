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

    # 必須カラムチェック
    required_cols = ['timestamp', 'enhanced_altitude', 'power', 'distance']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()

    df = df[required_cols].dropna()

    # 勾配計算
    df['delta_altitude'] = df['enhanced_altitude'].diff()
    df['delta_distance'] = df['distance'].diff()

    # 0距離差はNaNにして勾配計算時に除外
    df.loc[df['delta_distance'] == 0, 'delta_distance'] = np.nan
    df['gradient'] = df['delta_altitude'] / df['delta_distance']

    # 勾配分類
    def classify_segment(g):
        if pd.isna(g):
            return 'flat'  # NaNは平坦扱い
        elif g > 0.03:
            return 'uphill'
        elif g < -0.03:
            return 'downhill'
        else:
            return 'flat'

    df['segment'] = df['gradient'].apply(classify_segment)

    # power数値化＆0以下は除外
    df['power'] = pd.to_numeric(df['power'], errors='coerce')
    df = df.dropna(subset=['power'])
    df = df[df['power'] > 0].reset_index(drop=True)

    return df

if uploaded_file is not None:
    st.sidebar.subheader("Input your weight (kg)")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    if weight <= 0:
        st.error("Weight must be greater than zero.")
        st.stop()

    with st.spinner('Analyzing...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("Failed to parse the file or missing required columns.")
        st.stop()

    # W/kg計算
    df['w_per_kg'] = df['power'] / weight

    st.success("✅ Analysis completed!")

    st.subheader("📊 Data preview (first 100 rows)")
    st.dataframe(df.head(100))

    # 地形別平均W/kg表示
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    overall_mean_wkg = df['w_per_kg'].mean().round(2)
    
    # SeriesをDataFrameに変換し、「Overall」行を追加
    mean_wkg_df = mean_wkg.reset_index()
    mean_wkg_df = mean_wkg_df.rename(columns={'segment': 'Terrain', 'w_per_kg': 'Avg W/kg'})
    overall_row = pd.DataFrame({'Terrain': ['Overall'], 'Avg W/kg': [overall_mean_wkg]})
    
    mean_wkg_df = pd.concat([mean_wkg_df, overall_row], ignore_index=True)
    
    st.subheader("🧮 Average W/kg by Terrain (NRRS-P)")
    st.dataframe(mean_wkg_df)
    st.write(mean_wkg_df)

    # タイム経過を秒で計算（グラフのX軸に使う）
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].iloc[0]
    df['elapsed_sec'] = (df['timestamp'] - start_time).dt.total_seconds()

    # 地形別データ準備
    df_uphill = df[df['segment'] == 'uphill']
    df_flat = df[df['segment'] == 'flat']
    df_downhill = df[df['segment'] == 'downhill']

    def smooth(series, window=10):
        return series.rolling(window=window, min_periods=1).mean()

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
        if data.empty:
            st.write(f"No data for {terrain}")
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(data['elapsed_sec'], data['w_per_kg'], s=10, alpha=0.3, label='Raw Data')
        ax.plot(data['elapsed_sec'], smooth(data['w_per_kg']), color=colors[terrain], linewidth=2, label='Smoothed')
        ax.set_title(f"W/kg - {terrain.capitalize()}")
        ax.set_xlabel("Elapsed Time (sec)")
        ax.set_ylabel("W/kg")
        ax.legend()
        st.pyplot(fig)

    # CSVダウンロード
    st.subheader("📁 Export CSV")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data, file_name="nrrs_parsed.csv", mime="text/csv")
