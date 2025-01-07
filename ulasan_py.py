import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("ulasan_gojek.csv")

# Fungsi untuk menghitung rekomendasi berdasarkan kemiripan teks
def get_recommendations(input_text, top_n=5):
    # Vectorisasi teks menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['content'])
    
    # Vectorisasi input pengguna
    input_vector = tfidf_vectorizer.transform([input_text])
    
    # Hitung kemiripan kosinus antara input dan ulasan lainnya
    cosine_sim = cosine_similarity(input_vector, tfidf_matrix)
    
    # Ambil rekomendasi berdasarkan kemiripan
    sim_scores = list(enumerate(cosine_sim[0]))  # Ambil kemiripan dengan input
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # Skip the first item (itself)
    recommended_indices = [x[0] for x in sim_scores]

    return data.iloc[recommended_indices]

# Halaman aplikasi Streamlit
st.title("Rekomendasi Berdasarkan Ulasan Gojek")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Ulasan Anda")

# Pilihan untuk memilih versi aplikasi
app_versions = data['appVersion'].unique()
selected_version = st.sidebar.selectbox("Pilih Versi Aplikasi", app_versions)

# Menampilkan ulasan berdasarkan versi aplikasi yang dipilih
filtered_data = data[data['appVersion'] == selected_version]
st.write(f"Ulasan untuk Versi Aplikasi {selected_version}:")
st.write(filtered_data[['userName', 'content', 'score', 'at']])

# Input teks ulasan pengguna yang sudah dihilangkan
# user_input = st.sidebar.text_area("Tulis ulasan Anda:") # Bagian ini dihapus


