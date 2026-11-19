import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Deteksi Fraud Kartu Kredit", page_icon="💳")

FITUR = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def muat_model():
    return joblib.load("model_terbaik.pkl")


model = muat_model()


def prediksi(df_fitur, threshold):
    proba = model.predict_proba(df_fitur[FITUR])[:, 1]
    label = (proba >= threshold).astype(int)
    return proba, label


st.title("💳 Deteksi Fraud Transaksi Kartu Kredit")
st.write(
    "Aplikasi ini memakai model Random Forest (dengan preprocessing dan SMOTE "
    "di dalam pipeline) untuk memprediksi apakah sebuah transaksi berisiko fraud."
)

# =========================
# PROFILE
# =========================

st.sidebar.subheader("👤 Profile")
st.sidebar.image("profile.jpg", width=120)
st.sidebar.markdown("### Muhamad Sriraka Ramadhan")
st.sidebar.caption("Mahasiswa Teknik Informatika")

st.sidebar.divider()

st.sidebar.write("**NPM:** 202343500152")
st.sidebar.write("**Program Studi:** Teknik Informatika")
st.sidebar.write("**Mata Kuliah:** Machine Learning")
st.sidebar.write("**Dosen Pengampu:** Nurfidah Dwitiyanti, M.Si")

st.sidebar.divider()


# =========================
# PENGATURAN
# =========================

st.sidebar.header("Pengaturan")

mode = st.sidebar.radio(
    "Mode input",
    ["Contoh transaksi", "Upload CSV"],
    key="mode_input"
)

threshold = st.sidebar.slider(
    "Ambang batas fraud",
    0.1, 0.9, 0.5, 0.05,
    help="Makin rendah, makin banyak transaksi ditandai fraud (recall naik, false alarm juga naik)."
)


if mode == "Contoh transaksi":
    data = pd.read_csv("sample_transaksi.csv")
    idx = st.selectbox(
        "Pilih transaksi",
        data.index,
        format_func=lambda i: f"Transaksi #{i + 1} (Amount: {data.loc[i, 'Amount']:.2f})",
    )
    baris = data.loc[[idx]]
    st.write("Ringkasan transaksi:")
    st.dataframe(baris[["Time", "Amount"]], hide_index=True)

    if st.button("Prediksi"):
        proba, label = prediksi(baris, threshold)
        if label[0] == 1:
            st.error(f"Risiko TINGGI fraud (probabilitas: {proba[0]:.1%})")
        else:
            st.success(f"Risiko RENDAH fraud (probabilitas: {proba[0]:.1%})")
        if "Class_asli" in data.columns:
            asli = "Fraud" if data.loc[idx, "Class_asli"] == 1 else "Normal"
            st.caption(f"Label sebenarnya di dataset: {asli}")

else:
    st.write(
        "Upload CSV yang berisi kolom: Time, V1-V28, dan Amount "
        "(contoh format: sample_transaksi.csv)."
    )
    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df_up = pd.read_csv(file)
        kurang = [k for k in FITUR if k not in df_up.columns]
        if kurang:
            st.error(f"Kolom berikut belum ada: {kurang}")
        else:
            proba, label = prediksi(df_up, threshold)
            hasil = df_up.copy()
            hasil["Probabilitas_fraud"] = proba.round(4)
            hasil["Prediksi"] = np.where(label == 1, "FRAUD", "Normal")

            col1, col2 = st.columns(2)
            col1.metric("Jumlah transaksi", len(hasil))
            col2.metric("Terdeteksi fraud", int((label == 1).sum()))

            st.dataframe(
                hasil[["Time", "Amount", "Probabilitas_fraud", "Prediksi"]],
                hide_index=True,
            )
            st.download_button(
                "Download hasil (CSV)",
                hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )