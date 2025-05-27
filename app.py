import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P プロトタイプ v0.2")
st.markdown("FITファイルから出力・地形分類・NRRS-Pを一発解析")

# 体重入力
weight = st.number_input("🏋️‍♂️ 体重を入力してください（kg）", min_value=30.0, max_value=150.0, value=60.0)

# ファイルアップロード
uploaded_file = st.file_uploader("📂 FITファイルをアップロード", type=["fit"])

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
        st.error(f"必須カラムが足りません: {missing_cols}")
        return pd.DataFrame()

    # 欠損値除去 & 必要カラム抽出
    df = df[required_cols].dropna()

    # 勾配の計算
    df['delta_altitude'] = df['enhanced_altitude'].diff()
    df['delta_distance'] = df['distance'].diff()
    df['gradient'] = df['delta_altitude'] / df['delta_distance'].replace(0, np.nan)

    # 地形分類
    def classify_segment(g):
        if g > 0.03:
            return 'uphill'
        elif g < -0.03:
            return 'downhill'
        else:
            return 'flat'

    df['segment'] = df['gradient'].apply(classify_segment)

    # W/kg の算出
    df['w_per_kg'] = df['power'] / weight

    return df

if uploaded_file is not None:
    with st.spinner('解析中...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("解析に失敗しました。ファイルやカラムの内容を確認してください。")
        st.stop()

    st.success("✅ 解析完了！")

    # NRRS-P 表示
    st.subheader("🧮 地形別 平均 W/kg (NRRS-P)")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    # 地形別 Powerグラフ（分割表示）
    st.subheader("📈 地形別 Powerグラフ")
    for seg_type in ['uphill', 'flat', 'downhill']:
        seg = df[df['segment'] == seg_type].copy()
        if not seg.empty:
            seg['timestamp'] = pd.to_datetime(seg['timestamp'])
            seg['elapsed_time'] = (seg['timestamp'] - seg['timestamp'].iloc[0]).dt.total_seconds()
            st.markdown(f"### ⛰ {seg_type.capitalize()}セクション")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(seg['elapsed_time'], seg['power'], label=seg_type, color='tab:blue')
            ax.set_xlabel("Elapsed Time in Segment (sec)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"{seg_type.capitalize()} - Power推移")
            ax.grid(True)
            st.pyplot(fig)

    # CSV 出力
    st.subheader("📁 CSV出力")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("CSVとして保存", csv_data, file_name="nrrs_parsed.csv", mime="text/csv")
