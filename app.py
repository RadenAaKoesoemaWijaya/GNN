import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import IDSGNNModel, APILogAutoencoder, calculate_anomaly_threshold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
import re
import json
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add this code near the top of your app.py file, after the imports and before any other Streamlit code

# Set page configuration
st.set_page_config(page_title='GRAPHNET', layout='wide')

# Custom CSS styling
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #FFFFFF;
        color:rgb(8, 43, 27)
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1E3D2F !important;
        font-weight: bold !important;
    }
    
    h1 {
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #F5F5F5;
        border-right: 2px solid #D4AF37;
    }
    
    /* Button styling */
    .stButton>button {
        background-color:rgb(49, 136, 97);
        color: white;
        border: 2px solid #D4AF37;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #D4AF37;
        color:rgb(58, 196, 134);
        border: 2px solid #1E3D2F;
    }
    
    /* Metric styling */
    .css-1xarl3l, .css-1r6slb0 {
        background-color: #F5F5F5;
        border: 1px solid #D4AF37;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F5F5;
        color:rgb(64, 199, 138);
        border-radius: 5px 5px 0 0;
        border: 1px solid #D4AF37;
        border-bottom: none;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #F5F5F5 !important;
        color:rgb(72, 199, 142) !important;
        font-weight: bold;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 2px solid #F5F5F5 !important;
    }
    
    .dataframe th {
        background-color: #1E3D2F !important;
        color: white !important;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #F5F5F5;
    }
    
    /* Success/Error message styling */
    .stSuccess, .css-r0qwyj {
        background-color: rgba(30, 61, 47, 0.2);
        border: 1px solid #1E3D2F;
        color:rgb(82, 207, 151);
    }
    
    .stError, .css-r0qwyj {
        background-color: rgba(212, 175, 55, 0.2);
        border: 1px solid #D4AF37;
        color:rgb(75, 192, 139);
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: #F5F5F5 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #F5F5F5;
    }
    
    /* Checkbox styling */
    .stCheckbox [data-baseweb="checkbox"] {
        color:rgb(65, 182, 129);
    }
    
    /* Select box styling */
    .stSelectbox [data-baseweb="select"] {
        border-color: #D4AF37;
    }
    
    /* File uploader styling */
    .stFileUploader [data-baseweb="file-uploader"] {
        border-color: #D4AF37;
        background-color: #F5F5F5;
    }
    
    /* Custom container for important sections */
    .highlight-container {
        background-color: rgba(212, 175, 55, 0.1);
        border-left: 5px solid #D4AF37;
        padding: 20px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    
    /* Custom container for results */
    .results-container {
        background-color: rgba(30, 61, 47, 0.1);
        border-left: 5px solid #1E3D2F;
        padding: 20px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Add this function after your CSS styling

def custom_container(content, container_type="highlight"):
    """Create a custom styled container for content"""
    if container_type == "highlight":
        st.markdown(f'<div class="highlight-container">{content}</div>', unsafe_allow_html=True)
    elif container_type == "results":
        st.markdown(f'<div class="results-container">{content}</div>', unsafe_allow_html=True)

def make_json_serializable(obj):
        """Convert non-serializable objects to serializable types"""
        if isinstance(obj, pd.DataFrame):
            # Convert DataFrame to a format that's JSON serializable
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'dtype') and pd.api.types.is_object_dtype(obj.dtype):
            # Convert object dtype to strings
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

# Fungsi untuk menampilkan halaman utama
def show_home_page():
    st.title('GRAPHNET - API Log Anomaly Detection')
    st.markdown("""
    ## Selamat Datang di Aplikasi Deteksi Anomali Log API
    
    Aplikasi ini membantu Anda mendeteksi anomali dalam log API menggunakan teknik machine learning canggih, 
    termasuk kombinasi Autoencoder dan Graph Neural Networks (GNN).
    
    ### Apa yang ingin Anda lakukan?
    """)
    
    # Tampilkan opsi navigasi dengan kartu yang menarik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; height: 200px;">
            <h3>Pengumpulan Data</h3>
            <p>Upload data log API dan lakukan pra-pemrosesan data</p>
            <br/>
        </div>
        """, unsafe_allow_html=True)
        collect_button = st.button("Pengumpulan Data", key="collect")
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; height: 200px;">
            <h3>Pelatihan Model</h3>
            <p>Latih model Autoencoder dan GNN untuk deteksi anomali</p>
            <br/>
        </div>
        """, unsafe_allow_html=True)
        train_button = st.button("Pelatihan Model", key="train")
    
    with col3:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; height: 200px;">
            <h3>Deteksi Anomali</h3>
            <p>Gunakan model terlatih untuk mendeteksi anomali pada data baru</p>
            <br/>
        </div>
        """, unsafe_allow_html=True)
        detect_button = st.button("Deteksi Anomali", key="detect")
    
    if collect_button:
        st.session_state['page'] = 'collect'
        st.rerun()
    
    if train_button:
        st.session_state['page'] = 'train'
        st.rerun()
    
    if detect_button:
        # Cek apakah model sudah ada
        models_exist = os.path.exists("models/autoencoder.pt") and os.path.exists("models/gnn_model.pt")
        has_data = 'df_processed' in st.session_state
        
        if models_exist and has_data:
            st.session_state['page'] = 'detect'
            st.rerun()
        else:
            # Tampilkan pesan peringatan dengan opsi
            st.warning("⚠️ Persyaratan untuk deteksi anomali belum terpenuhi!")
            
            if not models_exist:
                st.info("🔧 Model belum dilatih. Anda perlu melatih model terlebih dahulu.")
                if st.button("Lanjut ke Pelatihan Model", key="train_from_detect"):
                    st.session_state['page'] = 'train'
                    st.rerun()
            
            if not has_data:
                st.info("📊 Data belum diproses. Anda perlu mengumpulkan dan memproses data terlebih dahulu.")
                if st.button("Mulai dari Pengumpulan Data", key="collect_from_detect"):
                    st.session_state['page'] = 'collect'
                    st.rerun()
            
            # Jika salah satu sudah ada, beri opsi yang sesuai
            if models_exist and not has_data:
                st.info("📤 Anda bisa upload data baru untuk dideteksi")
                st.session_state['page'] = 'detect'
                st.rerun()
            elif has_data and not models_exist:
                st.info("🎯 Gunakan data yang sudah diproses untuk melatih model")
                if st.button("Latih Model dengan Data Tersedia", key="train_with_existing"):
                    st.session_state['page'] = 'train'
                    st.rerun()
    
    # Tampilkan informasi tambahan tentang aplikasi
    st.markdown("""
    ---
    ### Tentang Aplikasi Ini
    
    Aplikasi Deteksi Anomali Log API ini memungkinkan Anda menganalisis log API, melakukan eksplorasi data, dan mendeteksi anomali menggunakan model machine learning.
    
    - Upload dataset log API (format CSV)
    - Lakukan pra-pemrosesan dan ekstraksi fitur
    - Latih model Autoencoder dan GNN untuk deteksi anomali
    - Deteksi anomali pada data baru
    - Visualisasikan hasil dan unduh laporan anomali
    
    ### Alur Kerja Sistem
    
    1. **Pengumpulan Data**: Log API dikumpulkan dan diproses
    2. **Pra-pemrosesan Data**: Data mentah diubah menjadi fitur yang dapat digunakan model
    3. **Pembentukan Grafik**: Data direpresentasikan sebagai grafik untuk analisis GNN
    4. **Pelatihan Model**: Model Autoencoder dan GNN dilatih dengan data normal
    5. **Deteksi Anomali**: Model terlatih digunakan untuk mendeteksi anomali pada data baru
    6. **Evaluasi Hasil**: Hasil deteksi dievaluasi dan ditampilkan
    
    Untuk petunjuk detail, lihat bagian bantuan atau dokumentasi.
    """)

# Fungsi untuk menampilkan halaman pengumpulan data
def show_data_collection_page():
    st.title("Pengumpulan Data Log API")
    
    st.markdown("""
    ### Tujuan: Mengumpulkan data mentah untuk analisis
    
    Upload file log API Anda dalam format CSV. Data yang dikumpulkan sebaiknya mencakup:
    - IP Address: Alamat asal permintaan
    - Timestamp: Waktu permintaan
    - User Agent: Informasi tentang perangkat atau aplikasi yang digunakan
    - Endpoint/API Call: API spesifik yang diakses
    - Parameters: Parameter yang dikirim dalam permintaan
    - Response Status Code: Kode status balasan (contoh: 200, 404)
    - Response Time: Waktu yang dibutuhkan untuk membalas permintaan
    """)
    
    # Upload file
    uploaded_file = st.file_uploader("Upload file log API (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Baca file CSV
            df = pd.read_csv(uploaded_file)
            
            # Tampilkan informasi dasar
            st.success(f"File berhasil diunggah! Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
            
            # Tampilkan sampel data
            st.subheader("Sampel Data")
            st.dataframe(df.head())
            
            # Tampilkan informasi kolom
            st.subheader("Informasi Kolom")
            col_info = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes,
                'Nilai Unik': [df[col].nunique() for col in df.columns],
                'Missing Values': df.isnull().sum().values,
                'Missing (%)': (df.isnull().sum() / len(df) * 100).values
            })
            st.dataframe(col_info)
            
            # Deteksi kolom yang diperlukan
            required_columns = ['ip_address', 'timestamp', 'user_agent', 'endpoint', 'parameters', 'status_code', 'response_time']
            missing_columns = [col for col in required_columns if not any(existing_col.lower().replace('_', '').startswith(col.replace('_', '')) for existing_col in df.columns)]
            
            if missing_columns:
                st.warning(f"Beberapa kolom yang direkomendasikan tidak ditemukan: {', '.join(missing_columns)}")
                
                # Opsi untuk memetakan kolom
                st.subheader("Pemetaan Kolom")
                st.markdown("Petakan kolom yang ada ke kolom yang diperlukan:")
                
                column_mapping = {}
                for req_col in required_columns:
                    if req_col in missing_columns:
                        column_mapping[req_col] = st.selectbox(
                            f"Pilih kolom untuk '{req_col}':",
                            options=["Tidak ada"] + list(df.columns),
                            key=f"map_{req_col}"
                        )
            
            # Simpan data dan lanjut ke preprocessing
            if st.button("Simpan dan Lanjutkan ke Pra-pemrosesan"):
                # Jika ada pemetaan kolom, terapkan
                if 'column_mapping' in locals() and column_mapping:
                    df_mapped = df.copy()
                    for req_col, mapped_col in column_mapping.items():
                        if mapped_col != "Tidak ada":
                            df_mapped[req_col] = df[mapped_col]
                    
                    # Simpan dataframe yang sudah dipetakan
                    df_mapped.to_csv("dataset_mapped.csv", index=False)
                    st.session_state['df'] = df_mapped
                else:
                    # Simpan dataframe asli
                    df.to_csv("dataset.csv", index=False)
                    st.session_state['df'] = df
                
                st.success("Data berhasil disimpan! Lanjut ke langkah pra-pemrosesan...")
                
                # Langsung navigasi ke halaman preprocessing
                st.session_state['page'] = 'preprocess'
                st.rerun()
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")

# Fungsi untuk pra-pemrosesan data log API
def preprocess_api_logs(df):
    """Pra-pemrosesan data log API untuk model deteksi anomali"""
    df = df.copy()
    
    # Step 1: Konversi timestamp ke format datetime
    timestamp_cols = [col for col in df.columns if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'timestamp'])]
    
    for col in timestamp_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            # Ekstrak fitur waktu
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            print(f"Converted {col} to datetime and extracted time features")
        except Exception as e:
            print(f"Failed to convert {col} to datetime: {str(e)}")
    
    # Step 2: Ekstraksi fitur dari IP address
    ip_cols = [col for col in df.columns if any(ip_keyword in col.lower() for ip_keyword in ['ip', 'address', 'src', 'source'])]
    
    for col in ip_cols:
        if df[col].dtype == 'object':
            # Cek apakah kolom berisi IP address
            if df[col].str.contains(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}').any():
                # Ekstrak oktet dari IP address
                try:
                    df[f'{col}_first_octet'] = df[col].str.extract(r'(\d{1,3})\.\d{1,3}\.\d{1,3}\.\d{1,3}').astype(float)
                    df[f'{col}_second_octet'] = df[col].str.extract(r'\d{1,3}\.(\d{1,3})\.\d{1,3}\.\d{1,3}').astype(float)
                    print(f"Extracted features from IP column: {col}")
                except Exception as e:
                    print(f"Failed to extract IP features from {col}: {str(e)}")
    
    # Step 3: Ekstraksi fitur dari User Agent
    ua_cols = [col for col in df.columns if any(ua_keyword in col.lower() for ua_keyword in ['user', 'agent', 'browser', 'ua'])]
    
    for col in ua_cols:
        if df[col].dtype == 'object':
            # Deteksi browser, OS, dan device
            df[f'{col}_is_mobile'] = df[col].str.contains('Mobile|Android|iOS', case=False).astype(int)
            df[f'{col}_is_chrome'] = df[col].str.contains('Chrome', case=False).astype(int)
            df[f'{col}_is_firefox'] = df[col].str.contains('Firefox', case=False).astype(int)
            df[f'{col}_is_safari'] = df[col].str.contains('Safari', case=False).astype(int)
            print(f"Extracted features from User Agent column: {col}")
    
    # Step 4: Ekstraksi fitur dari endpoint/API call
    endpoint_cols = [col for col in df.columns if any(ep_keyword in col.lower() for ep_keyword in ['endpoint', 'api', 'url', 'path'])]
    
    for col in endpoint_cols:
        if df[col].dtype == 'object':
            # One-hot encoding untuk endpoint
            endpoint_dummies = pd.get_dummies(df[col], prefix=f'{col}_endpoint')
            df = pd.concat([df, endpoint_dummies], axis=1)
            print(f"Applied one-hot encoding to endpoint column: {col}")
    
    # Step 5: Konversi status code ke kategori
    status_cols = [col for col in df.columns if any(status_keyword in col.lower() for status_keyword in ['status', 'code', 'response_code'])]
    
    for col in status_cols:
        # Kategorikan status code
        df[f'{col}_is_success'] = ((df[col] >= 200) & (df[col] < 300)).astype(int)
        df[f'{col}_is_redirect'] = ((df[col] >= 300) & (df[col] < 400)).astype(int)
        df[f'{col}_is_client_error'] = ((df[col] >= 400) & (df[col] < 500)).astype(int)
        df[f'{col}_is_server_error'] = (df[col] >= 500).astype(int)
        print(f"Categorized status codes from column: {col}")
    
    # Step 6: Agregasi fitur berdasarkan IP dan waktu
    # Hitung jumlah permintaan per IP dalam interval waktu tertentu
    if ip_cols and timestamp_cols:
        try:
            # Gunakan kolom IP dan timestamp pertama yang ditemukan
            ip_col = ip_cols[0]
            time_col = timestamp_cols[0]
            
            # Buat fitur agregat
            ip_counts = df.groupby(ip_col).size().reset_index(name='requests_per_ip')
            df = df.merge(ip_counts, on=ip_col, how='left')
            
            # Jika timestamp sudah dikonversi ke datetime
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                # Buat interval waktu (misalnya 5 menit)
                df['time_interval'] = df[time_col].dt.floor('5min')
                
                # Hitung permintaan per IP per interval waktu
                ip_time_counts = df.groupby([ip_col, 'time_interval']).size().reset_index(name='requests_per_ip_5min')
                df = df.merge(ip_time_counts, on=[ip_col, 'time_interval'], how='left')
                
                print("Created aggregated features based on IP and time intervals")
        except Exception as e:
            print(f"Failed to create aggregated features: {str(e)}")
    
    # Step 7: Hapus kolom yang tidak diperlukan untuk model
    # Hapus kolom original yang sudah diekstrak fiturnya
    cols_to_drop = []
    cols_to_drop.extend(timestamp_cols)  # Timestamp asli sudah diekstrak fiturnya
    cols_to_drop.extend(endpoint_cols)   # Endpoint sudah di-one-hot encoding
    cols_to_drop.extend(ua_cols)         # User agent sudah diekstrak fiturnya
    
    # Juga hapus kolom non-numerik lainnya yang tidak bisa digunakan model
    for col in df.columns:
        if df[col].dtype == 'object' and col not in cols_to_drop:
            cols_to_drop.append(col)
    
    # Hapus kolom 'time_interval' jika ada (hanya digunakan untuk agregasi)
    if 'time_interval' in df.columns:
        cols_to_drop.append('time_interval')
    
    # Hapus kolom yang sudah ditentukan
    df_numeric = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped {len(cols_to_drop)} columns that are not needed for modeling")
    
    # Step 8: Tangani missing values
    df_numeric = df_numeric.fillna(0)
    
    # Step 9: Normalisasi fitur numerik
    scaler = StandardScaler()
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    df_numeric[numeric_cols] = scaler.fit_transform(df_numeric[numeric_cols])
    
    return df_numeric, scaler, df

# Fungsi untuk membuat graph dari data log API
def create_api_log_graph(df, ip_col=None, endpoint_col=None):
    """Membuat graph dari data log API untuk GNN"""
    # Identifikasi kolom IP dan endpoint jika tidak ditentukan
    if ip_col is None:
        ip_cols = [str(col) for col in df.columns if any(ip_keyword in str(col).lower() for ip_keyword in ['ip', 'address', 'src', 'source'])]
        ip_col = ip_cols[0] if ip_cols else None
    
    if endpoint_col is None:
        endpoint_cols = [str(col) for col in df.columns if any(ep_keyword in str(col).lower() for ep_keyword in ['endpoint', 'api', 'url', 'path'])]
        endpoint_col = endpoint_cols[0] if endpoint_cols else None
    
    # Jika kolom IP atau endpoint tidak ditemukan, gunakan pendekatan sequential
    if ip_col is None or endpoint_col is None or ip_col not in df.columns or endpoint_col not in df.columns:
        print("IP or endpoint column not found, using sequential graph approach")
        # Buat graph berdasarkan urutan sekuensial
        edge_index = []
        for i in range(len(df) - 1):
            edge_index.append([i, i + 1])  # Hubungkan setiap baris dengan baris berikutnya
            edge_index.append([i + 1, i])  # Bidirectional
    else:
        print(f"Creating graph based on IP ({ip_col}) and endpoint ({endpoint_col}) relationships")
        # Buat graph berdasarkan hubungan IP-endpoint
        # Buat mapping untuk IP dan endpoint
        unique_ips = df[ip_col].unique()
        unique_endpoints = df[endpoint_col].unique()
        
        ip_to_idx = {ip: i for i, ip in enumerate(unique_ips)}
        endpoint_to_idx = {endpoint: i + len(unique_ips) for i, endpoint in enumerate(unique_endpoints)}
        
        # Buat edge index
        edge_index = []
        for _, row in df.iterrows():
            ip_idx = ip_to_idx[row[ip_col]]
            endpoint_idx = endpoint_to_idx[row[endpoint_col]]
            
            # Hubungkan IP ke endpoint
            edge_index.append([ip_idx, endpoint_idx])
            edge_index.append([endpoint_idx, ip_idx])  # Bidirectional
    
    # Konversi ke tensor PyTorch
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Fitur node adalah fitur numerik dari dataframe
    x = torch.tensor(df.select_dtypes(include=[np.number]).values, dtype=torch.float)
    
    # Buat objek Data PyTorch Geometric
    data = Data(x=x, edge_index=edge_index)
    
    return data

# Fungsi untuk menampilkan halaman pelatihan model
def show_training_page():
    st.title("Pelatihan Model")
    
    st.markdown("""
    ### Tujuan: Melatih model Autoencoder dan GNN dengan data normal
    
    Pada langkah ini, kita akan melatih dua model:
    1. **Autoencoder**: Untuk mendeteksi anomali berdasarkan fitur tabular
    2. **Graph Neural Network (GNN)**: Untuk mendeteksi anomali berbasis grafik
    """)
    
    # Cek apakah data sudah diproses
    if 'df_processed' not in st.session_state:
        st.warning("Data belum diproses. Silakan kembali ke langkah Pengumpulan Data dan Pra-pemrosesan.")
        if st.button("Kembali ke Pengumpulan Data"):
            st.session_state['page'] = 'collect'
            st.rerun()
        return
    
    # Ambil data yang sudah diproses
    df_processed = st.session_state['df_processed']
    
    # Tampilkan informasi data yang akan digunakan untuk pelatihan
    st.subheader("Data untuk Pelatihan")
    st.write(f"Jumlah sampel: {df_processed.shape[0]}")
    st.write(f"Jumlah fitur: {df_processed.shape[1]}")
    
    # Opsi pelatihan
    st.subheader("Opsi Pelatihan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Opsi untuk Autoencoder
        st.markdown("#### Opsi Autoencoder")
        
        hidden_dims = st.text_input("Hidden Dimensions (pisahkan dengan koma)", "128,64,32")
        hidden_dims = [int(dim.strip()) for dim in hidden_dims.split(",")]
        
        dropout_ae = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
        learning_rate_ae = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        
        epochs_ae = st.slider("Epochs", 10, 200, 50, 10)
    
    with col2:
        # Opsi untuk GNN
        st.markdown("#### Opsi GNN")
        
        hidden_channels = st.slider("Hidden Channels", 16, 256, 64, 16)
        num_layers = st.slider("Number of Layers", 1, 5, 2, 1)
        dropout_gnn = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1, key="dropout_gnn")
        learning_rate_gnn = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            key="lr_gnn"
        )
        
        epochs_gnn = st.slider("Epochs", 10, 200, 50, 10, key="epochs_gnn")
    
    # Opsi pembagian data
    st.subheader("Pembagian Data")
    train_size = st.slider("Proporsi Data Training", 0.5, 0.9, 0.8, 0.05)
    
    # State untuk tracking progress training
    if 'training_step' not in st.session_state:
        st.session_state['training_step'] = 'autoencoder'
    
    # Tombol untuk memulai pelatihan Autoencoder
    st.subheader("🎯 Pelatihan Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Autoencoder Training")
        if st.button("Mulai Pelatihan Autoencoder", type="primary", key="train_ae"):
            st.session_state['training_step'] = 'autoencoder'
            st.session_state['start_autoencoder_training'] = True
            st.rerun()
    
    with col2:
        st.markdown("### GNN Training")
        gnn_ready = st.session_state.get('autoencoder_trained', False)
        if st.button("Mulai Pelatihan GNN", type="primary", key="train_gnn", disabled=not gnn_ready):
            st.session_state['training_step'] = 'gnn'
            st.session_state['start_gnn_training'] = True
            st.rerun()
    
    # Pelatihan Autoencoder
    if st.session_state.get('start_autoencoder_training', False):
        with st.spinner("Melatih Autoencoder..."):
            try:
                # Persiapan data
                X = df_processed.values
                X_train, X_test = train_test_split(X, train_size=train_size, random_state=42)
                
                input_dim = X_train.shape[1]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Inisialisasi model
                autoencoder = APILogAutoencoder(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout_ae
                ).to(device)
                
                optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate_ae)
                
                # Konversi data ke tensor
                X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
                
                # Training loop dengan visualisasi
                train_losses_ae = []
                test_losses_ae = []
                
                progress_bar_ae = st.progress(0)
                status_text_ae = st.empty()
                
                for epoch in range(epochs_ae):
                    autoencoder.train()
                    optimizer_ae.zero_grad()
                    
                    outputs = autoencoder(X_train_tensor)
                    loss = F.mse_loss(outputs, X_train_tensor)
                    loss.backward()
                    optimizer_ae.step()
                    
                    train_losses_ae.append(loss.item())
                    
                    autoencoder.eval()
                    with torch.no_grad():
                        test_outputs = autoencoder(X_test_tensor)
                        test_loss = F.mse_loss(test_outputs, X_test_tensor)
                        test_losses_ae.append(test_loss.item())
                    
                    progress_bar_ae.progress((epoch + 1) / epochs_ae)
                    status_text_ae.text(f"Autoencoder - Epoch {epoch+1}/{epochs_ae}, Loss: {loss.item():.6f}")
                
                # Hitung threshold
                autoencoder.eval()
                with torch.no_grad():
                    test_outputs = autoencoder(X_test_tensor)
                    reconstruction_errors = F.mse_loss(test_outputs, X_test_tensor, reduction='none').mean(dim=1).cpu().numpy()
                
                threshold = calculate_anomaly_threshold(reconstruction_errors, method='std')
                
                # Visualisasi hasil training
                st.success(f"✅ Pelatihan Autoencoder selesai! Threshold: {threshold:.6f}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Training curve
                ax1.plot(train_losses_ae, label='Train Loss', color='blue')
                ax1.plot(test_losses_ae, label='Test Loss', color='orange')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Autoencoder Training Curve')
                ax1.legend()
                ax1.grid(True)
                
                # Distribution of reconstruction errors
                ax2.hist(reconstruction_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
                ax2.set_xlabel('Reconstruction Error')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Reconstruction Errors')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Simpan model dan parameter
                os.makedirs("models", exist_ok=True)
                torch.save(autoencoder.state_dict(), "models/autoencoder.pt")
                with open("models/anomaly_threshold.txt", "w") as f:
                    f.write(str(threshold))
                
                autoencoder_params = {
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout_ae
                }
                with open("models/autoencoder_params.json", "w") as f:
                    json.dump(autoencoder_params, f)
                
                # Update session state
                st.session_state['autoencoder'] = autoencoder
                st.session_state['anomaly_threshold'] = threshold
                st.session_state['autoencoder_trained'] = True
                st.session_state['start_autoencoder_training'] = False
                
                # Reset untuk GNN training
                st.session_state['start_gnn_training'] = False
                
                st.info("🎯 Autoencoder siap! Sekarang Anda bisa melatih GNN.")
                
            except Exception as e:
                st.error(f"Error saat pelatihan Autoencoder: {str(e)}")
                st.session_state['start_autoencoder_training'] = False
    
    # Pelatihan GNN (setelah Autoencoder selesai)
    if st.session_state.get('start_gnn_training', False) and st.session_state.get('autoencoder_trained', False):
        with st.spinner("Melatih GNN..."):
            try:
                # Persiapan data untuk GNN
                X = df_processed.values
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Buat graph data
                graph_data = create_api_log_graph(pd.DataFrame(X))
                
                # Inisialisasi model GNN
                num_features = graph_data.x.shape[1]
                num_classes = 2
                
                gnn_model = IDSGNNModel(
                    num_features=num_features,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    dropout=dropout_gnn
                ).to(device)
                
                # Include reconstruction head parameters in optimizer
                gnn_params = list(gnn_model.parameters())
                if hasattr(gnn_model, 'reconstruction_head'):
                    gnn_params.extend(list(gnn_model.reconstruction_head.parameters()))
                optimizer_gnn = torch.optim.Adam(gnn_params, lr=learning_rate_gnn)
                
                # Training loop dengan visualisasi
                train_losses_gnn = []
                
                progress_bar_gnn = st.progress(0)
                status_text_gnn = st.empty()
                
                # Untuk GNN, kita gunakan data yang sama sebagai target (unsupervised)
                # Dalam konteks deteksi anomali, kita latih untuk merekonstruksi fitur
                for epoch in range(epochs_gnn):
                    gnn_model.train()
                    optimizer_gnn.zero_grad()
                    
                    # For GNN training - use the GNN model without global pooling
                    # Create a dummy batch tensor for compatibility, but we'll use node-level outputs
                    batch = torch.zeros(graph_data.x.size(0), dtype=torch.long).to(device)
                    
                    # Get node-level embeddings (before global pooling)
                    x = graph_data.x.to(device)
                    edge_index = graph_data.edge_index.to(device)
                    
                    # Process through GAT layers to get node embeddings
                    for gat_layer in gnn_model.gat_layers:
                        x = gat_layer(x, edge_index)
                        x = F.elu(x)
                        x = F.dropout(x, p=gnn_model.dropout, training=True)
                    
                    # Simple reconstruction from node embeddings
                    if not hasattr(gnn_model, 'reconstruction_head'):
                        gnn_model.reconstruction_head = torch.nn.Linear(x.size(-1), graph_data.x.size(1)).to(device)
                        
                    reconstructed = gnn_model.reconstruction_head(x)
                    loss = F.mse_loss(reconstructed, graph_data.x.to(device))
                    loss.backward()
                    optimizer_gnn.step()
                    
                    train_losses_gnn.append(loss.item())
                    
                    progress_bar_gnn.progress((epoch + 1) / epochs_gnn)
                    status_text_gnn.text(f"GNN - Epoch {epoch+1}/{epochs_gnn}, Loss: {loss.item():.6f}")
                
                # Visualisasi hasil training
                st.success("✅ Pelatihan GNN selesai!")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses_gnn, label='GNN Training Loss', color='purple')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('GNN Training Curve')
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Simpan model GNN
                torch.save(gnn_model.state_dict(), "models/gnn_model.pt")
                
                gnn_params = {
                    "num_features": num_features,
                    "num_classes": num_classes,
                    "hidden_channels": hidden_channels,
                    "num_layers": num_layers,
                    "dropout": dropout_gnn
                }
                with open("models/gnn_params.json", "w") as f:
                    json.dump(gnn_params, f)
                
                # Update session state
                st.session_state['gnn_model'] = gnn_model
                st.session_state['gnn_trained'] = True
                st.session_state['start_gnn_training'] = False
                
                # Tombol untuk lanjut ke deteksi
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Lanjut ke Deteksi Anomali", type="primary"):
                        st.session_state['page'] = 'detect'
                        st.rerun()
                with col2:
                    if st.button("🏠 Kembali ke Beranda"):
                        st.session_state['page'] = 'home'
                        st.rerun()
                
            except Exception as e:
                st.error(f"Error saat pelatihan GNN: {str(e)}")
                st.session_state['start_gnn_training'] = False

# Fungsi untuk menampilkan halaman deteksi anomali
def show_detection_page():
    st.title("Langkah 4: Deteksi Anomali")
    
    st.markdown("""
    ### Tujuan: Menggunakan model yang sudah dilatih untuk mendeteksi anomali pada data baru
    
    Upload data log API baru untuk dianalisis, atau gunakan data test yang sudah ada.
    """)
    
    # Cek apakah model sudah dilatih
    models_exist = os.path.exists("models/autoencoder.pt") and os.path.exists("models/gnn_model.pt")
    has_trained_models = 'autoencoder' in st.session_state and 'gnn_model' in st.session_state
    
    if not models_exist and not has_trained_models:
        st.error("❌ Model belum tersedia untuk deteksi anomali")
        st.info("Anda perlu melatih model terlebih dahulu sebelum dapat mendeteksi anomali.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Pelatihan Model", type="primary"):
                st.session_state['page'] = 'train'
                st.rerun()
        with col2:
            if st.button("🏠 Kembali ke Beranda"):
                st.session_state['page'] = 'home'
                st.rerun()
        return
    
    # Opsi untuk menggunakan data test atau upload data baru
    data_option = st.radio(
        "Pilih sumber data untuk deteksi anomali:",
        options=["Upload data baru", "Gunakan data test yang sudah ada"]
    )
    
    if data_option == "Upload data baru":
        # Upload file
        uploaded_file = st.file_uploader("Upload file log API (CSV)", type=["csv"], key="detect_upload")
        
        if uploaded_file is not None:
            try:
                # Baca file CSV
                df = pd.read_csv(uploaded_file)
                
                # Tampilkan informasi dasar
                st.success(f"File berhasil diunggah! Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
                
                # Tampilkan sampel data
                st.subheader("Sampel Data")
                st.dataframe(df.head())
                
                # Simpan dataframe ke session state
                st.session_state['df_detect'] = df
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
    
    else:  # Gunakan data test yang sudah ada
        if 'df_processed' in st.session_state:
            # Gunakan sebagian data yang sudah diproses sebagai data test
            df_processed = st.session_state['df_processed']
            
            # Ambil sampel acak dari data
            sample_size = min(1000, len(df_processed))
            df_sample = df_processed.sample(n=sample_size, random_state=42)
            
            st.success(f"Menggunakan {sample_size} sampel acak dari data yang sudah diproses")
            
            # Tampilkan sampel data
            st.subheader("Sampel Data")
            st.dataframe(df_sample.head())
            
            # Simpan dataframe ke session state
            st.session_state['df_detect'] = df_sample
        else:
            st.warning("Data yang sudah diproses tidak ditemukan. Silakan upload data baru.")
    
    # Tombol untuk mendeteksi anomali
    if 'df_detect' in st.session_state and st.button("Deteksi Anomali"):
        with st.spinner("Mendeteksi anomali... Ini mungkin memerlukan waktu beberapa menit."):
            try:
                # Ambil data untuk deteksi
                df_detect = st.session_state['df_detect']
                
                # Pra-pemrosesan data
                if data_option == "Upload data baru":
                    # Pra-pemrosesan data baru
                    df_processed, scaler, df_original = preprocess_api_logs(df_detect)
                else:
                    # Data sudah diproses
                    df_processed = df_detect
                    df_original = df_detect
                
                # Load model jika belum ada di session state
                if 'autoencoder' not in st.session_state:
                    # Load parameter model
                    with open("models/autoencoder_params.json", "r") as f:
                        autoencoder_params = json.load(f)
                    
                    with open("models/gnn_params.json", "r") as f:
                        gnn_params = json.load(f)
                    
                    # Load threshold
                    with open("models/anomaly_threshold.txt", "r") as f:
                        threshold = float(f.read())
                    
                    # Buat model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    autoencoder = APILogAutoencoder(
                        input_dim=autoencoder_params["input_dim"],
                        hidden_dims=autoencoder_params["hidden_dims"],
                        dropout=autoencoder_params["dropout"]
                    ).to(device)
                    
                    gnn_model = IDSGNNModel(
                        num_features=gnn_params["num_features"],
                        num_classes=gnn_params["num_classes"],
                        hidden_channels=gnn_params["hidden_channels"],
                        num_layers=gnn_params["num_layers"],
                        dropout=gnn_params["dropout"]
                    ).to(device)
                    
                    # Load state dict
                    autoencoder.load_state_dict(torch.load("models/autoencoder.pt"))
                    gnn_model.load_state_dict(torch.load("models/gnn_model.pt"))
                    
                    # Simpan ke session state
                    st.session_state['autoencoder'] = autoencoder
                    st.session_state['gnn_model'] = gnn_model
                    st.session_state['anomaly_threshold'] = threshold
                else:
                    # Ambil model dari session state
                    autoencoder = st.session_state['autoencoder']
                    gnn_model = st.session_state['gnn_model']
                    threshold = st.session_state['anomaly_threshold']
                
                # Deteksi anomali dengan Autoencoder
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                X = torch.tensor(df_processed.values, dtype=torch.float).to(device)
                
                autoencoder.eval()
                with torch.no_grad():
                    outputs = autoencoder(X)
                    reconstruction_errors = F.mse_loss(outputs, X, reduction='none').mean(dim=1).cpu().numpy()
                
                # Identifikasi anomali berdasarkan threshold
                anomalies = reconstruction_errors > threshold
                
                # Tambahkan hasil ke dataframe original
                df_original['reconstruction_error'] = reconstruction_errors
                df_original['is_anomaly'] = anomalies
                df_original['anomaly_score'] = reconstruction_errors / threshold
                
                # Tampilkan hasil
                st.subheader("Hasil Deteksi Anomali")
                
                # Statistik hasil
                num_anomalies = anomalies.sum()
                anomaly_percentage = (num_anomalies / len(df_original)) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Data", len(df_original))
                
                with col2:
                    st.metric("Jumlah Anomali", int(num_anomalies))
                
                with col3:
                    st.metric("Persentase Anomali", f"{anomaly_percentage:.2f}%")
                
                # Tampilkan data dengan anomali
                st.subheader("Data dengan Anomali")
                anomaly_df = df_original[df_original['is_anomaly']].sort_values('anomaly_score', ascending=False)
                st.dataframe(anomaly_df)
                
                # Visualisasi distribusi reconstruction error
                st.subheader("Distribusi Reconstruction Error")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sns.histplot(reconstruction_errors, bins=50, kde=True, ax=ax)
                ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
                ax.set_xlabel('Reconstruction Error')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Reconstruction Errors')
                ax.legend()
                
                st.pyplot(fig)
                
                # Visualisasi anomali berdasarkan waktu (jika ada kolom timestamp)
                timestamp_cols = [col for col in df_original.columns if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'timestamp'])]
                
                if timestamp_cols:
                    st.subheader("Anomali Berdasarkan Waktu")
                    time_col = timestamp_cols[0]
                    
                    try:
                        # Konversi ke datetime jika belum
                        if not pd.api.types.is_datetime64_any_dtype(df_original[time_col]):
                            df_original[time_col] = pd.to_datetime(df_original[time_col])
                        
                        # Plot anomali berdasarkan waktu
                        fig = px.scatter(
                            df_original,
                            x=time_col,
                            y='anomaly_score',
                            color='is_anomaly',
                            title="Anomaly Score Over Time",
                            labels={'anomaly_score': 'Anomaly Score', time_col: 'Time'},
                            color_discrete_map={True: 'red', False: 'blue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Tidak dapat membuat visualisasi berdasarkan waktu: {str(e)}")
                
                # Opsi untuk mengunduh hasil
                csv = df_original.to_csv(index=False)
                st.download_button(
                    label="Unduh Hasil Deteksi Anomali",
                    data=csv,
                    file_name="anomaly_detection_results.csv",
                    mime="text/csv"
                )
                
                # Simpan hasil ke session state
                st.session_state['detection_results'] = df_original
                
                # Tambahkan tombol untuk lanjut ke evaluasi hasil
                if st.button("Lanjut ke Evaluasi Hasil"):
                    st.session_state['page'] = 'evaluate'
                    st.rerun()
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mendeteksi anomali: {str(e)}")
                st.exception(e)

# Fungsi untuk menampilkan halaman evaluasi hasil
def show_evaluation_page():
    st.title("Penggabungan Hasil dan Evaluasi")
    
    st.markdown("""
    ### Tujuan: Mengevaluasi hasil deteksi anomali dan memperbaiki model
    
    Pada langkah ini, kita akan mengevaluasi hasil deteksi anomali dan memberikan umpan balik untuk perbaikan model.
    """)
    
    # Cek apakah hasil deteksi sudah ada
    if 'detection_results' not in st.session_state:
        st.warning("Hasil deteksi anomali belum ada. Silakan kembali ke langkah Deteksi Anomali.")
        if st.button("Kembali ke Deteksi Anomali"):
            st.session_state['page'] = 'detect'
            st.rerun()
        return
    
    # Ambil hasil deteksi
    results = st.session_state['detection_results']
    
    # Tampilkan ringkasan hasil
    st.subheader("Ringkasan Hasil Deteksi")
    
    # Statistik hasil
    num_anomalies = results['is_anomaly'].sum()
    anomaly_percentage = (num_anomalies / len(results)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data", len(results))
    
    with col2:
        st.metric("Jumlah Anomali", int(num_anomalies))
    
    with col3:
        st.metric("Persentase Anomali", f"{anomaly_percentage:.2f}%")
    
    # Tampilkan top anomali
    st.subheader("Top 10 Anomali Berdasarkan Score")
    top_anomalies = results[results['is_anomaly']].sort_values('anomaly_score', ascending=False).head(10)
    st.dataframe(top_anomalies)
    
    # Opsi untuk validasi manual
    st.subheader("Validasi Manual Anomali")
    st.markdown("""
    Validasi hasil deteksi anomali untuk meningkatkan akurasi model di masa depan.
    Pilih beberapa anomali terdeteksi dan tentukan apakah itu benar-benar anomali atau false positive.
    """)
    
    # Pilih anomali untuk validasi
    if num_anomalies > 0:
        anomaly_indices = results[results['is_anomaly']].index.tolist()
        selected_indices = st.multiselect(
            "Pilih anomali untuk validasi:",
            options=anomaly_indices,
            format_func=lambda x: f"Index {x} (Score: {results.loc[x, 'anomaly_score']:.4f})"
        )
        
        if selected_indices:
            # Tampilkan data anomali yang dipilih
            st.dataframe(results.loc[selected_indices])
            
            # Form validasi
            validation_results = {}
            
            for idx in selected_indices:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"Anomali Index {idx} (Score: {results.loc[idx, 'anomaly_score']:.4f})")
                
                with col2:
                    is_true_anomaly = st.radio(
                        "Apakah ini benar-benar anomali?",
                        options=["Ya", "Tidak"],
                        key=f"validate_{idx}"
                    )
                    validation_results[idx] = is_true_anomaly == "Ya"
            
            # Tombol untuk menyimpan validasi
            if st.button("Simpan Hasil Validasi"):
                # Hitung metrik berdasarkan validasi
                true_positives = sum(1 for idx, is_true in validation_results.items() if is_true)
                false_positives = sum(1 for idx, is_true in validation_results.items() if not is_true)
                
                precision = true_positives / len(validation_results) if len(validation_results) > 0 else 0
                
                st.success("Hasil validasi berhasil disimpan!")
                
                # Tampilkan metrik
                st.subheader("Metrik Berdasarkan Validasi")
                st.write(f"True Positives: {true_positives}")
                st.write(f"False Positives: {false_positives}")
                st.write(f"Precision: {precision:.4f}")
                
                # Simpan hasil validasi ke file
                validation_df = pd.DataFrame({
                    'index': list(validation_results.keys()),
                    'is_true_anomaly': list(validation_results.values())
                })
                
                validation_df.to_csv("validation_results.csv", index=False)
                
                # Simpan ke session state
                st.session_state['validation_results'] = validation_results
        else:
            st.info("Pilih anomali untuk validasi")
    else:
        st.info("Tidak ada anomali terdeteksi untuk divalidasi")
    
    # Rekomendasi untuk perbaikan model
    st.subheader("Rekomendasi untuk Perbaikan Model")
    
    st.markdown("""
    Berdasarkan hasil deteksi dan validasi, berikut adalah beberapa rekomendasi untuk meningkatkan akurasi model:
    
    1. **Penyesuaian Threshold**: Jika terlalu banyak false positives, pertimbangkan untuk meningkatkan threshold. Jika terlalu banyak false negatives, pertimbangkan untuk menurunkan threshold.
    
    2. **Fitur Tambahan**: Pertimbangkan untuk menambahkan fitur baru yang dapat membantu membedakan antara perilaku normal dan anomali.
    
    3. **Pelatihan Ulang**: Latih ulang model dengan data yang lebih banyak dan lebih representatif.
    
    4. **Ensemble Method**: Pertimbangkan untuk menggabungkan hasil dari beberapa model deteksi anomali untuk meningkatkan akurasi.
    """)
    
    # Opsi untuk penyesuaian threshold
    st.subheader("Penyesuaian Threshold")
    
    current_threshold = st.session_state.get('anomaly_threshold', 0.1)
    new_threshold = st.slider("Threshold Baru", 0.0, 1.0, float(current_threshold), 0.01)
    
    if st.button("Terapkan Threshold Baru"):
        # Hitung ulang anomali dengan threshold baru
        results['is_anomaly_new'] = results['reconstruction_error'] > new_threshold
        results['anomaly_score_new'] = results['reconstruction_error'] / new_threshold
        
        # Tampilkan perbandingan
        num_anomalies_new = results['is_anomaly_new'].sum()
        anomaly_percentage_new = (num_anomalies_new / len(results)) * 100
        
        st.subheader("Perbandingan Hasil dengan Threshold Baru")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Jumlah Anomali (Threshold Lama)", int(num_anomalies))
            st.metric("Persentase Anomali (Threshold Lama)", f"{anomaly_percentage:.2f}%")
        
        with col2:
            st.metric("Jumlah Anomali (Threshold Baru)", int(num_anomalies_new))
            st.metric("Persentase Anomali (Threshold Baru)", f"{anomaly_percentage_new:.2f}%")
        
        # Simpan threshold baru
        st.session_state['anomaly_threshold'] = new_threshold
        
        # Simpan ke file
        with open("models/anomaly_threshold.txt", "w") as f:
            f.write(str(new_threshold))
        
        st.success(f"Threshold berhasil diubah menjadi {new_threshold}!")

# Fungsi untuk menampilkan halaman pra-pemrosesan
def show_preprocessing_page():
    st.title("Pra-pemrosesan Data dan Pembentukan Fitur")
    
    st.markdown("""
    ### Tujuan: Mengubah data mentah menjadi format yang dapat digunakan oleh model
    
    Pada langkah ini, kita akan melakukan pra-pemrosesan data log API dan membentuk fitur untuk model Autoencoder dan GNN dengan visualisasi detail setiap tahapnya.
    """)
    
    # Cek apakah data sudah diupload
    if 'df' not in st.session_state:
        st.warning("Data belum diupload. Silakan kembali ke langkah Pengumpulan Data.")
        if st.button("Kembali ke Pengumpulan Data"):
            st.session_state['page'] = 'collect'
            st.rerun()
        return
    
    # Ambil data yang sudah diupload
    df = st.session_state['df'].copy()
    
    # Tampilkan informasi dasar
    st.subheader("Data yang Akan Diproses")
    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
    
    # Tampilkan sampel data
    st.subheader("Sampel Data Mentah")
    st.dataframe(df.head())
    
    # Analisis tipe data dan missing values
    st.subheader("Analisis Data Awal")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Baris", df.shape[0])
        st.metric("Total Kolom", df.shape[1])
    
    with col2:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Values (%)", f"{missing_pct:.2f}%")
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        st.metric("Duplikat (%)", f"{duplicate_pct:.2f}%")
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Kolom Numerik", numeric_cols)
        st.metric("Kolom Kategorikal", categorical_cols)
    
    # Visualisasi distribusi tipe data
    st.subheader("Distribusi Tipe Data")
    col_types = df.dtypes.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    col_types.plot(kind='bar', ax=ax)
    ax.set_title("Distribusi Tipe Data Kolom")
    ax.set_xlabel("Tipe Data")
    ax.set_ylabel("Jumlah Kolom")
    st.pyplot(fig)
    
    # Heatmap missing values
    st.subheader("Peta Missing Values")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, ax=ax, cmap='viridis')
    ax.set_title("Visualisasi Missing Values")
    st.pyplot(fig)
    
    # Opsi pra-pemrosesan
    st.subheader("Opsi Pra-pemrosesan")
    
    # Deteksi kolom timestamp, IP, dan endpoint
    timestamp_cols = [col for col in df.columns if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'timestamp'])]
    ip_cols = [col for col in df.columns if any(ip_keyword in col.lower() for ip_keyword in ['ip', 'address', 'src', 'source'])]
    endpoint_cols = [col for col in df.columns if any(ep_keyword in col.lower() for ep_keyword in ['endpoint', 'api', 'url', 'path'])]
    
    # Tampilkan kolom yang terdeteksi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Kolom Timestamp Terdeteksi:")
        if timestamp_cols:
            st.write(timestamp_cols)
        else:
            st.write("Tidak ditemukan")
    
    with col2:
        st.write("Kolom IP Terdeteksi:")
        if ip_cols:
            st.write(ip_cols)
        else:
            st.write("Tidak ditemukan")
    
    with col3:
        st.write("Kolom Endpoint Terdeteksi:")
        if endpoint_cols:
            st.write(endpoint_cols)
        else:
            st.write("Tidak ditemukan")
    
    # Opsi untuk memilih kolom yang akan digunakan
    st.subheader("Pilih Kolom untuk Diproses")
    
    # Pilih kolom timestamp
    selected_timestamp = None
    if timestamp_cols:
        selected_timestamp = st.selectbox("Pilih Kolom Timestamp", timestamp_cols)
    
    # Pilih kolom IP
    selected_ip = None
    if ip_cols:
        selected_ip = st.selectbox("Pilih Kolom IP", ip_cols)
    
    # Pilih kolom endpoint
    selected_endpoint = None
    if endpoint_cols:
        selected_endpoint = st.selectbox("Pilih Kolom Endpoint", endpoint_cols)
    
    # Simpan kolom yang dipilih ke session state
    st.session_state['selected_timestamp'] = selected_timestamp
    st.session_state['selected_ip'] = selected_ip
    st.session_state['selected_endpoint'] = selected_endpoint
    
    # Tombol untuk melanjutkan ke langkah berikutnya
    if 'preprocessing_step' not in st.session_state:
        st.session_state['preprocessing_step'] = 0
    
    if st.session_state['preprocessing_step'] == 0:
        if st.button("Mulai Pra-pemrosesan - Tahap 1"):
            st.session_state['preprocessing_step'] = 1
            st.rerun()
    
    # Inisialisasi status dan progress bar
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Container untuk visualisasi tiap tahap
    if st.session_state['preprocessing_step'] == 1:
        st.subheader("📊 Tahap 1: Ekstraksi Fitur dari Timestamp")
        
        status_text.text("Tahap 1: Ekstraksi Fitur dari Timestamp...")
        progress_bar.progress(15)
            
        if selected_timestamp:
            original_col = df[selected_timestamp]
            
            # Konversi timestamp
            try:
                df[selected_timestamp] = pd.to_datetime(df[selected_timestamp])
                
                # Visualisasi perubahan tipe data
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Tipe Data Awal:", original_col.dtype)
                    st.write("Contoh data:", original_col.iloc[0])
                with col2:
                    st.write("Tipe Data Baru:", df[selected_timestamp].dtype)
                    st.write("Contoh data:", df[selected_timestamp].iloc[0])
                
                # Ekstrak fitur waktu
                df[f'{selected_timestamp}_hour'] = df[selected_timestamp].dt.hour
                df[f'{selected_timestamp}_day'] = df[selected_timestamp].dt.day
                df[f'{selected_timestamp}_dayofweek'] = df[selected_timestamp].dt.dayofweek
                
                # Visualisasi distribusi waktu
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Distribusi jam
                df[f'{selected_timestamp}_hour'].hist(bins=24, ax=axes[0])
                axes[0].set_title("Distribusi Jam")
                axes[0].set_xlabel("Jam")
                
                # Distribusi hari dalam bulan
                df[f'{selected_timestamp}_day'].hist(bins=31, ax=axes[1])
                axes[1].set_title("Distribusi Hari dalam Bulan")
                axes[1].set_xlabel("Hari")
                
                # Distribusi hari dalam minggu
                df[f'{selected_timestamp}_dayofweek'].hist(bins=7, ax=axes[2])
                axes[2].set_title("Distribusi Hari dalam Minggu")
                axes[2].set_xlabel("Hari (0=Senin)")
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Gagal mengkonversi timestamp: {str(e)}")
        else:
            st.warning("Tidak ada kolom timestamp yang dipilih")
        
        # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step1'] = df.copy()
        
        if st.button("Lanjut ke Tahap 2"):
            st.session_state['preprocessing_step'] = 2
            st.rerun()
    
    elif st.session_state['preprocessing_step'] == 2:
        st.subheader("🔍 Tahap 2: Ekstraksi Fitur dari IP Address")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step1']
        
        status_text.text("Tahap 2: Ekstraksi Fitur dari IP Address...")
        progress_bar.progress(30)
        
        if selected_ip and selected_ip in df.columns:
            original_col = df[selected_ip]
            
            # Ekstrak fitur IP
            try:
                df[f'{selected_ip}_first_octet'] = df[selected_ip].str.extract(r'(\d{1,3})\.\d{1,3}\.\d{1,3}\.\d{1,3}').astype(float)
                df[f'{selected_ip}_second_octet'] = df[selected_ip].str.extract(r'\d{1,3}\.(\d{1,3})\.\d{1,3}\.\d{1,3}').astype(float)
                
                # Visualisasi distribusi oktet IP
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Distribusi first octet
                df[f'{selected_ip}_first_octet'].hist(bins=20, ax=axes[0])
                axes[0].set_title("Distribusi First Octet IP")
                axes[0].set_xlabel("First Octet")
                
                # Distribusi second octet
                df[f'{selected_ip}_second_octet'].hist(bins=20, ax=axes[1])
                axes[1].set_title("Distribusi Second Octet IP")
                axes[1].set_xlabel("Second Octet")
                
                st.pyplot(fig)
                
                # Tampilkan statistik
                st.write("Statistik IP Address:")
                ip_stats = pd.DataFrame({
                    'First Octet': [df[f'{selected_ip}_first_octet'].mean(), df[f'{selected_ip}_first_octet'].std()],
                    'Second Octet': [df[f'{selected_ip}_second_octet'].mean(), df[f'{selected_ip}_second_octet'].std()]
                }, index=['Mean', 'Std'])
                st.dataframe(ip_stats)
                
            except Exception as e:
                st.error(f"Gagal mengekstrak fitur IP: {str(e)}")
        else:
            st.warning("Tidak ada kolom IP yang dipilih")
        
        # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step2'] = df.copy()
        
        if st.button("Lanjut ke Tahap 3"):
            st.session_state['preprocessing_step'] = 3
            st.rerun()
    
    elif st.session_state['preprocessing_step'] == 3:
        st.subheader("🌐 Tahap 3: Ekstraksi Fitur User Agent")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step2']
        
        status_text.text("Tahap 3: Ekstraksi Fitur User Agent...")
        progress_bar.progress(40)
        
        ua_cols = [col for col in df.columns if any(ua_keyword in col.lower() for ua_keyword in ['user', 'agent', 'browser', 'ua'])]
        
        if ua_cols:
            ua_col = ua_cols[0]  # Gunakan kolom pertama
            
            # Ekstrak fitur User Agent
            df[f'{ua_col}_is_mobile'] = df[ua_col].str.contains('Mobile|Android|iOS', case=False).astype(int)
            df[f'{ua_col}_is_chrome'] = df[ua_col].str.contains('Chrome', case=False).astype(int)
            df[f'{ua_col}_is_firefox'] = df[ua_col].str.contains('Firefox', case=False).astype(int)
            df[f'{ua_col}_is_safari'] = df[ua_col].str.contains('Safari', case=False).astype(int)
            
            # Visualisasi distribusi browser
            browser_counts = pd.DataFrame({
                'Mobile': [df[f'{ua_col}_is_mobile'].sum()],
                'Chrome': [df[f'{ua_col}_is_chrome'].sum()],
                'Firefox': [df[f'{ua_col}_is_firefox'].sum()],
                'Safari': [df[f'{ua_col}_is_safari'].sum()]
            }).T
            
            fig, ax = plt.subplots(figsize=(10, 4))
            browser_counts.plot(kind='bar', ax=ax)
            ax.set_title("Distribusi Tipe Browser dari User Agent")
            ax.set_ylabel("Jumlah")
            ax.set_xlabel("Tipe Browser")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Tampilkan persentase
            st.write("Persentase Tipe Browser:")
            total_rows = len(df)
            browser_pct = pd.DataFrame({
                'Jumlah': [df[f'{ua_col}_is_mobile'].sum(), df[f'{ua_col}_is_chrome'].sum(), 
                          df[f'{ua_col}_is_firefox'].sum(), df[f'{ua_col}_is_safari'].sum()],
                'Persentase (%)': [(df[f'{ua_col}_is_mobile'].sum()/total_rows)*100, 
                                  (df[f'{ua_col}_is_chrome'].sum()/total_rows)*100,
                                  (df[f'{ua_col}_is_firefox'].sum()/total_rows)*100,
                                  (df[f'{ua_col}_is_safari'].sum()/total_rows)*100]
            }, index=['Mobile', 'Chrome', 'Firefox', 'Safari'])
            st.dataframe(browser_pct)
            
        else:
            st.warning("Tidak ada kolom User Agent yang ditemukan")
        
        # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step3'] = df.copy()
        
        if st.button("Lanjut ke Tahap 4"):
            st.session_state['preprocessing_step'] = 4
            st.rerun()
    
    elif st.session_state['preprocessing_step'] == 4:
        st.subheader("🔗 Tahap 4: One-hot Encoding untuk Endpoint")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step3']
        
        status_text.text("Tahap 4: One-hot Encoding Endpoint...")
        progress_bar.progress(55)
        
        if selected_endpoint and selected_endpoint in df.columns:
            # One-hot encoding
            endpoint_dummies = pd.get_dummies(df[selected_endpoint], prefix=f'{selected_endpoint}_endpoint')
            
            # Visualisasi endpoint yang paling sering muncul
            endpoint_counts = df[selected_endpoint].value_counts().head(10)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Top 10 endpoint
            endpoint_counts.plot(kind='bar', ax=ax1)
            ax1.set_title("Top 10 Endpoint yang Paling Sering Muncul")
            ax1.set_xlabel("Endpoint")
            ax1.set_ylabel("Frekuensi")
            plt.xticks(rotation=45)
            
            # Distribusi one-hot encoding
            if len(endpoint_dummies.columns) <= 20:  # Tampilkan hanya jika tidak terlalu banyak
                endpoint_dummies.sum().plot(kind='bar', ax=ax2)
                ax2.set_title("Distribusi One-hot Encoded Features")
                ax2.set_xlabel("Feature")
                ax2.set_ylabel("Jumlah True")
                plt.xticks(rotation=45)
            else:
                ax2.text(0.5, 0.5, f"Terdapat {len(endpoint_dummies.columns)}\nfitur one-hot encoding", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title("Jumlah Fitur One-hot Encoding")
            
            st.pyplot(fig)
            
            # Gabungkan ke dataframe utama
            df = pd.concat([df, endpoint_dummies], axis=1)
            
            st.write(f"Jumlah fitur baru dari one-hot encoding: {len(endpoint_dummies.columns)}")
            
        else:
            st.warning("Tidak ada kolom endpoint yang dipilih")

            # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step4'] = df.copy()
        
        if st.button("Lanjut ke Tahap 5"):
            st.session_state['preprocessing_step'] = 5
            st.rerun()

    elif st.session_state['preprocessing_step'] == 5:
        st.subheader("🔗 Tahap 5: Kategorisasi Status Code")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step4']        

        # Tahap 5: Kategorisasi status code
        status_text.text("Tahap 5: Kategorisasi Status Code...")
        progress_bar.progress(70)
            
        with st.container():
                st.subheader("📊 Tahap 5: Kategorisasi Status Code")
                
                status_cols = [col for col in df.columns if any(status_keyword in col.lower() for status_keyword in ['status', 'code', 'response_code'])]
                
                if status_cols:
                    status_col = status_cols[0]
                    
                    # Kategorikan status code
                    df[f'{status_col}_is_success'] = ((df[status_col] >= 200) & (df[status_col] < 300)).astype(int)
                    df[f'{status_col}_is_redirect'] = ((df[status_col] >= 300) & (df[status_col] < 400)).astype(int)
                    df[f'{status_col}_is_client_error'] = ((df[status_col] >= 400) & (df[status_col] < 500)).astype(int)
                    df[f'{status_col}_is_server_error'] = (df[status_col] >= 500).astype(int)
                    
                    # Visualisasi distribusi kategori
                    status_categories = pd.DataFrame({
                        'Success (2xx)': [df[f'{status_col}_is_success'].sum()],
                        'Redirect (3xx)': [df[f'{status_col}_is_redirect'].sum()],
                        'Client Error (4xx)': [df[f'{status_col}_is_client_error'].sum()],
                        'Server Error (5xx)': [df[f'{status_col}_is_server_error'].sum()]
                    }).T
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Pie chart
                    ax1.pie(status_categories.values.flatten(), labels=status_categories.index, autopct='%1.1f%%')
                    ax1.set_title("Distribusi Kategori Status Code")
                    
                    # Bar chart
                    status_categories.plot(kind='bar', ax=ax2)
                    ax2.set_title("Jumlah per Kategori Status Code")
                    ax2.set_ylabel("Jumlah")
                    ax2.set_xlabel("Kategori")
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                    
                    # Tampilkan distribusi status code asli
                    st.write("Distribusi Status Code Asli (Top 10):")
                    status_dist = df[status_col].value_counts().head(10)
                    st.dataframe(status_dist)
                    
                else:
                    st.warning("Tidak ada kolom status code yang ditemukan")
        
        # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step5'] = df.copy()
        
        if st.button("Lanjut ke Tahap 6"):
            st.session_state['preprocessing_step'] = 6
            st.rerun()
    
    elif st.session_state['preprocessing_step'] == 6:
        st.subheader("📈 Tahap 6: Agregasi Fitur Berdasarkan IP dan Waktu")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step5']
        
        status_text.text("Tahap 6: Agregasi Fitur...")
        progress_bar.progress(85)
        
        if selected_ip and selected_timestamp and selected_ip in df.columns and selected_timestamp in df.columns:
            try:
                # Agregasi berdasarkan IP
                ip_counts = df.groupby(selected_ip).size().reset_index(name='requests_per_ip')
                
                # Visualisasi distribusi permintaan per IP
                fig, axes = plt.subplots(1, 2, figsize=(15, 4))
                
                # Top 20 IP dengan permintaan terbanyak
                top_ips = ip_counts.sort_values('requests_per_ip', ascending=False).head(20)
                top_ips.plot(x=selected_ip, y='requests_per_ip', kind='bar', ax=axes[0])
                axes[0].set_title("Top 20 IP dengan Permintaan Terbanyak")
                axes[0].set_xlabel("IP Address")
                axes[0].set_ylabel("Jumlah Permintaan")
                plt.xticks(rotation=45)
                
                # Distribusi permintaan per IP
                ip_counts['requests_per_ip'].hist(bins=50, ax=axes[1])
                axes[1].set_title("Distribusi Jumlah Permintaan per IP")
                axes[1].set_xlabel("Jumlah Permintaan")
                axes[1].set_ylabel("Frekuensi")
                
                st.pyplot(fig)
                
                # Gabungkan kembali ke dataframe
                df = df.merge(ip_counts, on=selected_ip, how='left')
                
                st.write(f"Jumlah IP unik: {len(ip_counts)}")
                st.write(f"Rata-rata permintaan per IP: {ip_counts['requests_per_ip'].mean():.2f}")
                
            except Exception as e:
                st.error(f"Gagal melakukan agregasi: {str(e)}")
        else:
            st.warning("IP atau timestamp tidak tersedia untuk agregasi")
        
        # Simpan state untuk tahap berikutnya
        st.session_state['df_after_step6'] = df.copy()
        
        if st.button("Lanjut ke Tahap 7"):
            st.session_state['preprocessing_step'] = 7
            st.rerun()
    
    elif st.session_state['preprocessing_step'] == 7:
        st.subheader("🔧 Tahap 7: Final Preprocessing")
        
        # Gunakan data dari tahap sebelumnya
        df = st.session_state['df_after_step6']
        
        status_text.text("Tahap 7: Final Preprocessing...")
        progress_bar.progress(95)
        
        # Identifikasi kolom untuk dihapus
        cols_to_drop = []
        cols_to_drop.extend([col for col in df.columns if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'timestamp'])])
        cols_to_drop.extend([col for col in df.columns if any(ep_keyword in col.lower() for ep_keyword in ['endpoint', 'api', 'url', 'path'])])
        cols_to_drop.extend([col for col in df.columns if any(ua_keyword in col.lower() for ua_keyword in ['user', 'agent', 'browser', 'ua'])])
        
        # Hapus kolom non-numerik
        for col in df.columns:
            if df[col].dtype == 'object' and col not in cols_to_drop:
                cols_to_drop.append(col)
        
        # Hapus kolom yang tidak diperlukan
        df_numeric = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Tangani missing values
        df_numeric = df_numeric.fillna(0)
        
        # Normalisasi fitur numerik
        scaler = StandardScaler()
        numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
        df_normalized = df_numeric.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(df_numeric[numeric_cols])
        
        # Visualisasi perubahan distribusi sebelum dan sesudah normalisasi
        st.subheader("Perbandingan Sebelum dan Sesudah Normalisasi")
        
        if len(numeric_cols) > 0:
            # Pilih 4 fitur pertama untuk visualisasi
            features_to_show = numeric_cols[:4]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(features_to_show):
                if i < len(axes):
                    axes[i].hist(df_numeric[feature], bins=30, alpha=0.7, label='Sebelum Normalisasi', color='blue')
                    axes[i].hist(df_normalized[feature], bins=30, alpha=0.7, label='Setelah Normalisasi', color='red')
                    axes[i].set_title(f"Distribusi {feature}")
                    axes[i].legend()
            
            st.pyplot(fig)
        
            # Simpan hasil ke session state
            st.session_state['df_processed'] = df_normalized
            st.session_state['preprocessing_complete'] = True
            
            st.success("✅ Pra-pemrosesan selesai!")
            
            # Tombol untuk melanjutkan
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Lanjut ke Pelatihan Model", type="primary"):
                    st.session_state['page'] = 'train'
                    st.rerun()
                    
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Navigation
    if st.session_state['page'] == 'home':
        show_home_page()
    elif st.session_state['page'] == 'collect':
        show_data_collection_page()
    elif st.session_state['page'] == 'preprocess':
        show_preprocessing_page()
    elif st.session_state['page'] == 'train':
        show_training_page()
    elif st.session_state['page'] == 'detect':
        show_detection_page()
    elif st.session_state['page'] == 'evaluate':
        show_evaluation_page()
    
# Run the app
if __name__ == '__main__':
    main()


