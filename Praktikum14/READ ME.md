# Deteksi Fraud Transaksi Kartu Kredit

Mini project end-to-end Machine Learning (Bab 14): perbandingan model, tuning, dan deployment.

## Tujuan
Membandingkan tiga algoritma untuk mendeteksi transaksi fraud pada data yang sangat tidak seimbang, memilih model terbaik, dan men-deploy-nya sebagai aplikasi web Streamlit.

## Dataset
Credit Card Fraud Detection (Machine Learning Group - ULB), Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

File `creditcard.csv` tidak disertakan karena ukurannya besar (sekitar 150 MB). Unduh dari link di atas dan taruh di folder proyek.

## Metode
- Preprocessing: hapus duplikat, RobustScaler untuk Amount dan Time
- Imbalance: SMOTE di dalam pipeline (hanya pada data training)
- Model: Logistic Regression, Random Forest, Histogram-based Gradient Boosting
- Tuning: RandomizedSearchCV, Stratified 3-Fold CV, scoring PR-AUC
- Evaluasi: Precision, Recall, F1, PR-AUC pada data test

## Hasil (data test)
| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0,398 | 0,800 | 0,531 | 0,679 |
| Random Forest | 0,789 | 0,789 | 0,789 | 0,805 |
| Gradient Boosting | 0,733 | 0,779 | 0,755 | 0,802 |

Model terpilih: Random Forest.

## Cara menjalankan
```
pip install -r requirements.txt
streamlit run app.py
```

## Isi folder
- `mini_project_fraud.ipynb`: notebook (EDA sampai evaluasi)
- `app.py`: aplikasi Streamlit
- `model_terbaik.pkl`: pipeline model terlatih
- `sample_transaksi.csv`: contoh transaksi untuk demo
- `requirements.txt`: daftar library

## Pembuat
[Muhamad sriraka ramadhan (202343500152)]