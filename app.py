import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P プロトタイプ v0.2")
st.markdown("FITファイルから出力・地形分類・NRRS-P指標を一発解析")

# =============================
# 体重入力（W/kg算出のため）
# =============================
weight = st.number_input("⚖️ 体重（kg）を入力", min_value=30.0, max_value=120.0, value=60.0, step=0.5)

# =============================
# ファイルアップロード
# =============================
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

    # 必要なカラムだけ抽出し、欠損値は削除
    df = df[required_cols].dropna()

    # 勾配計算
    df['delta_altitude'] = df['enhanced_altitude'].diff()
    df['delta_distance'] = df['distance'].diff()
    df['gradient'] = df['delta_altitude'] / df['delta_distance'].replace(0, np.nan)

    # 勾配で地形分類
    def classify_segment(g):
        if g > 0.03:
            return 'uphill'
        elif g < -0.03:
            return 'downhill'
        else:
            return 'flat'

    df['segment'] = df['gradient'].apply(classify_segment)

    return df

# =============================
# 解析実行
# =============================
if uploaded_file is not None:
    with st.spinner('解析中...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("解析に失敗しました。ファイルやカラムの内容を確認してください。")
        st.stop()

    # W/kg の計算
    df['w_per_kg'] = df['power'] / weight

    st.success("✅ 解析完了！")

    # カラム一覧表示
    st.subheader("=== カラム一覧 ===")
    st.write(df.columns.tolist())

    # データの先頭を表示
    st.subheader("📊 データ表示（先頭100件）")
    st.dataframe(df.head(100))

    # Powerと地形セグメントの可視化
    st.subheader("📈 出力（Power）と地形セグメントの可視化")
    fig, ax = plt.subplots(figsize=(10, 4))
    for seg_type in ['uphill', 'flat', 'downhill']:
        seg = df[df['segment'] == seg_type]
        ax.plot(seg['timestamp'], seg['power'], label=seg_type)
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.legend()
    st.pyplot(fig)

    # 地形別 W/kg 平均表示（NRRS-P）
    st.subheader("🧮 地形別 平均 W/kg (NRRS-P)")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    # CSVダウンロードボタン
    st.subheader("📁 CSV出力")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("CSVとして保存", csv_data, file_name="nprs_parsed.csv", mime="text/csv")
