import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NPRS-P プロトタイプ v0.1")
st.markdown("FITファイルから出力・地形分類を一発解析")

uploaded_file = st.file_uploader("📂 FITファイルをアップロード", type=["fit"])

def parse_fit_to_df(fit_file):
    records = []

    with fitdecode.FitReader(fit_file) as fit:
        for frame in fit:
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
                record = {field.name: field.value for field in frame.fields}
                records.append(record)

    df = pd.DataFrame(records)
    df = df[['timestamp', 'altitude', 'power', 'distance']].dropna()
    df['delta_altitude'] = df['altitude'].diff()
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
    with st.spinner('解析中...'):
        df = parse_fit_to_df(uploaded_file)
        st.success("✅ 解析完了！")

        st.subheader("📊 データ表示")
        st.dataframe(df.head(100))

        st.subheader("📈 出力（Power）と地形セグメントの可視化")
        fig, ax = plt.subplots(figsize=(10, 4))
        for seg_type in ['uphill', 'flat', 'downhill']:
            seg = df[df['segment'] == seg_type]
            ax.plot(seg['timestamp'], seg['power'], label=seg_type)
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (W)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📁 CSV出力")
        st.download_button("CSVとして保存", df.to_csv(index=False).encode(), file_name="nprs_parsed.csv", mime="text/csv")
