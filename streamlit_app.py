import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi bantu konversi teks menjadi angka
def convert_k(x):
    try:
        x = str(x).lower().replace("favorite", "").replace("(", "").replace(")", "").strip()
        if 'k' in x:
            return float(x.replace('k', '')) * 1000
        return float(x)
    except:
        return np.nan

# Load dan bersihkan data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("20240121_shopee_sample_data (1).csv")

    df['total_sold'] = df['total_sold'].apply(convert_k)
    df['total_rating'] = df['total_rating'].apply(convert_k)
    df['favorite'] = df['favorite'].apply(convert_k)
    df['item_rating'] = pd.to_numeric(df['item_rating'], errors='coerce')

    features = ['price_ori', 'price_actual', 'item_rating', 'total_rating', 'favorite']
    df_num = df[features].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled, columns=features)

    return df_num, scaled_df, scaler

# Load data
df_num, scaled_df, scaler = load_and_prepare_data()

# Train model
reg_model = LinearRegression()
reg_model.fit(scaled_df, df_num['price_actual'])

kmeans_model = KMeans(n_clusters=3, random_state=42)
df_num['cluster'] = kmeans_model.fit_predict(scaled_df)

# UI Streamlit
st.title("\U0001F4E6 Prediksi & Segmentasi Produk Shopee")

st.sidebar.header("Masukkan Fitur Produk")

input_data = {
    'price_ori': st.sidebar.number_input("Price ori", min_value=0.0, value=20.0),
    'price_actual': st.sidebar.number_input("Price actual", min_value=0.0, value=15.0),
    'item_rating': st.sidebar.number_input("Item rating", min_value=0.0, max_value=5.0, value=4.5),
    'total_rating': st.sidebar.number_input("Total rating", min_value=0.0, value=100.0),
    'favorite': st.sidebar.number_input("Favorite", min_value=0.0, value=50.0)
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Prediksi dan cluster
predicted_price = reg_model.predict(input_scaled)[0]
predicted_cluster = kmeans_model.predict(input_scaled)[0]

# Output
st.subheader("\U0001F4C8 Prediksi Harga Akhir")
st.success(f"Hasil prediksi harga jual produk: **RM {predicted_price:.2f}**")

st.subheader("\U0001F9E0 Segmentasi Produk")
st.info(f"Produk Anda termasuk dalam **Cluster {predicted_cluster}**")

# Deskripsi cluster
cluster_description = {
    0: "\U0001F4E6 Produk umum dengan performa rata-rata",
    1: "\U0001F525 Produk populer (rating & favorit tinggi)",
    2: "\U0001F9CA Produk kurang laku (rating/favorite rendah)"
}
desc = cluster_description.get(predicted_cluster, "Deskripsi tidak tersedia.")
st.info(f"Deskripsi Cluster: {desc}")

# Distribusi Cluster
st.subheader("\U0001F4CA Proporsi Produk per Cluster (%)")
cluster_counts = df_num['cluster'].value_counts().sort_index()
cluster_dist = df_num['cluster'].value_counts(normalize=True).sort_index() * 100
for i, pct in cluster_dist.items():
    st.write(f"Cluster {i}: {pct:.2f}% ({int(cluster_counts[i])} produk)")

# Visualisasi PCA 2D
st.subheader("\U0001F9ED Visualisasi Cluster dalam 2D (PCA)")
pca = PCA(n_components=2)
cluster_2d = pca.fit_transform(scaled_df)
df_plot = pd.DataFrame(cluster_2d, columns=["PC1", "PC2"])
df_plot['cluster'] = df_num['cluster']

fig, ax = plt.subplots()
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="cluster", palette="Set2", ax=ax)
ax.set_title("Visualisasi Clustering (PCA)")
st.pyplot(fig)

# Skenario uji cepat
if st.sidebar.button("Gunakan Skenario: Produk Laris"):
    st.session_state['price_ori'] = 100.0
    st.session_state['price_actual'] = 80.0
    st.session_state['item_rating'] = 4.9
    st.session_state['total_rating'] = 5000
    st.session_state['favorite'] = 2000
    st.rerun()
