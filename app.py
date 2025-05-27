import streamlit as st
import pandas as pd
import numpy as np
import fitdecode
import matplotlib.pyplot as plt

st.title("🏃‍♂️ NRRS-P プロトタイプ v0.3")
st.markdown("FITファイルから出力・地形分類・能力評価まで一発解析")

uploaded_file = st.file_uploader("📂 FITファイルをアップロード", type=["fit"])

# ----------------------------
# FITファイル解析関数
# ----------------------------
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

    df = df[required_cols].dropna()

    # 勾配計算と地形分類
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

# ----------------------------
# メイン処理
# ----------------------------
if uploaded_file is not None:
    with st.spinner('解析中...'):
        df = parse_fit_to_df(uploaded_file)

    if df.empty:
        st.error("解析に失敗しました。ファイルやカラムの内容を確認してください。")
        st.stop()

    st.success("✅ 解析完了！")

    # ユーザー体重入力
    st.subheader("⚖️ 体重を入力してください (kg)")
    weight = st.number_input("体重 (kg)", min_value=30.0, max_value=120.0, value=60.0, step=0.5)

    # W/kg列を追加
    df['w_per_kg'] = df['power'] / weight

    # Power=0を除外（各種処理の前にやる）
    df = df[df['power'] > 0]

    if df.empty:
        st.warning("出力（Power）が0のデータしか存在しませんでした。")
        st.stop()

    # カラム一覧表示
    st.subheader("📋 カラム一覧")
    st.write(df.columns.tolist())

    # データ表示
    st.subheader("📊 データ（先頭100件）")
    st.dataframe(df.head(100))

    # 地形別W/kg平均（Power=0は除外済み）
    st.subheader("🧮 地形別 平均 W/kg (NRRS-P)")
    mean_wkg = df.groupby('segment')['w_per_kg'].mean().round(2)
    st.write(mean_wkg)

    # 地形別にグラフ分割
    st.subheader("📈 地形別 Power 時系列グラフ")
    for seg_type in ['uphill', 'flat', 'downhill']:
        seg = df[df['segment'] == seg_type].copy()
        if seg.empty:
            continue
        # 累積時間を算出
        seg['elapsed_time'] = (seg['timestamp'] - seg['timestamp'].iloc[0]).dt.total_seconds()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(seg['elapsed_time'], seg['power'], label=f"{seg_type}")
        ax.set_title(f"{seg_type.capitalize()} セグメント")
        ax.set_xlabel("経過時間 (秒)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        st.pyplot(fig)

    # CSV出力
    st.subheader("📁 CSV出力")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("CSVとして保存", csv_data, file_name="nrrs_p_cleaned.csv", mime="text/csv")
