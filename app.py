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
    
    st.write("=== カラム一覧 ===")
    st.write(df.columns.tolist())
    
    # ここで一旦止めてカラムを確認するためにreturn
    return df
    
    # 以下は本来の処理
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
