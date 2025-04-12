import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from gensim.models import Word2Vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Fungsi rata-rata vektor kata (Word2Vec)
def get_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Inisialisasi riwayat
if "history" not in st.session_state:
    st.session_state.history = []

# ======== LOAD SEMUA MODEL ========
model_w2v = Word2Vec.load("models/model_w2v.model")
model_cnn_lstm = load_model("models/cnn_lstm_model_v2.h5")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model_nb = joblib.load("models/model_naive_bayes.pkl")
model_lr = joblib.load("models/model_logistic_regression.pkl")
model_mlp = joblib.load("models/model_mlp.pkl")

# ======== UI STREAMLIT ========
st.set_page_config(page_title="Sentimen OVO", layout="centered")
tab1, tab2 = st.tabs(["🔍 Prediksi Sentimen", "📋 Riwayat"])

# ======== TAB 1: PREDIKSI ========
with tab1:
    st.title("🔍 Analisis Sentimen Ulasan Aplikasi OVO")
    st.write("Masukkan ulasan dan pilih model untuk melihat sentimen:")

    text_input = st.text_area("💬 Tulis ulasan Anda di sini:")

    # Pilihan model
    model_choice = st.radio("Pilih Model:", ("CNN-LSTM", "Naive Bayes", "Logistic Regression", "MLP"))

    if st.button("Prediksi Sentimen"):
        if text_input:
            if model_choice in ["CNN-LSTM", "MLP"]:
                if model_choice == "CNN-LSTM":
                    # Word2Vec + reshape untuk CNN-LSTM
                    vec = get_sentence_vector(text_input, model_w2v).reshape(1, -1, 1)
                    prediction = model_cnn_lstm.predict(vec)
                    label = "Positif ✅" if prediction[0][0] > 0.5 else "Negatif ❌"

                elif model_choice == "MLP":
                    # Word2Vec tanpa reshape channel
                    vec = get_sentence_vector(text_input, model_w2v).reshape(1, -1)
                    prediction = model_mlp.predict(vec)
                    label = "Positif ✅" if prediction[0] > 0.5 else "Negatif ❌"
            else:
                # Gunakan TF-IDF untuk Naive Bayes dan Logistic Regression
                X_tfidf = tfidf_vectorizer.transform([text_input])
                if model_choice == "Naive Bayes":
                    prediction = model_nb.predict(X_tfidf)
                elif model_choice == "Logistic Regression":
                    prediction = model_lr.predict(X_tfidf)
                label = "Positif ✅" if prediction[0] == 1 else "Negatif ❌"

            st.subheader("Hasil Prediksi:")
            st.success(f"Sentimen: {label}")

            # Simpan ke riwayat
            st.session_state.history.append((text_input, label, model_choice))

            # WordCloud
            st.subheader("🔠 WordCloud dari Ulasan Anda")
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text_input)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

# ======== TAB 2: HISTORY ========
with tab2:
    st.title("📋 Riwayat Prediksi")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history, columns=["Komentar", "Sentimen", "Model"])
        st.table(df_history)

        csv = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="riwayat_sentimen.csv", mime="text/csv")
    else:
        st.info("Belum ada riwayat prediksi.")
