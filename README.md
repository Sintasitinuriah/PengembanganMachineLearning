# 📱 Analisis dan Klasifikasi Ulasan Aplikasi OVO

Proyek ini bertujuan untuk mengumpulkan, menganalisis, dan mengklasifikasikan ulasan pengguna dari aplikasi OVO menggunakan metode scraping dan machine learning berbasis deep learning. Percobaan prediksinya bisa diakses di sini ya: 
[Analisis Senitimen Ulasan Aplikasi OVO](https://predictsentimenovoreview.streamlit.app/)

## 📌 Deskripsi Proyek

- Menggunakan teknik web scraping untuk mengambil ulasan dari Google Play Store terkait aplikasi OVO.
- Melakukan preprocessing data teks: tokenisasi, stopword removal, dan normalisasi.
- Representasi teks menggunakan Word2Vec dan TF-IDF.
- Membangun model klasifikasi menggunakan CNN-LSTM, Naive Bayes, Logistic Regression, MLP, CNN-LSTM-RandomSearch.
- Optimasi hyperparameter dengan Keras Tuner.
- Evaluasi model menggunakan metrik akurasi, confusion matrix, dan classification report.

## 📁 Struktur Proyek

```
├── models/                               # models yang dihasilkan dan digunakan pada predict_model
├── reviews_ovo.csv                       # Dataset hasil scraping
├── Scrapper_Aplikasi_OVO_PlayStore.ipynb # Kode Scrapping ulasan playstore 
├── analisis-sentiment-review-ovo.ipynb   # Notebook utama
├── requirements.txt                      # Dependensi proyek
├── README.md                             # Dokumentasi proyek ini
```

## ⚙️ Instalasi

1. Clone repository:
```bash
git clone https://github.com/Sintasitinuriah/PengembanganMachineLearning.git
cd PengembanganMachineLearning
```

2. (Opsional) Buat virtual environment:
```bash
python -m venv env
source env/bin/activate  # Mac/Linux
env\Scripts\activate     # Windows
```

3. Install dependensi:
```bash
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

1. Pastikan kamu sudah memiliki dataset hasil scraping di folder utama
2. Jalankan notebook:
```bash
jupyter notebook analisis-sentiment-review-ovo.ipynb
```

## 🧪 Evaluasi Model

- Model: CNN + LSTM, MLP, Naive Bayes, Logistic Regression, best CNN + LSTM + Keras tuner RandomSearch
- Input: Word2Vec Vectorized Text, TF-IDF
- Output: Klasifikasi ulasan (Positif/Negatif)
- Evaluasi:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - Visualisasi hasil training (akurasi & loss)

## 🔧 Dependensi Utama

- `tensorflow`
- `scikit-learn`
- `gensim`
- `keras-tuner`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

## 👤 Author
  
[GitHub](https://github.com/Sintasitinuriah) | [LinkedIn](https://www.linkedin.com/in/sinta-siti-nuriah/)
