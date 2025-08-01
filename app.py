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
from model import IDSGNNModel
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
import re
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
st.set_page_config(page_title='Anomaly Detection Webapp', layout='wide')

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
    st.title('Anomaly Detection Webapp')
    st.markdown("""
    ## Welcome to the Anomaly Detection Webapp
    
    This application helps you detect anomalies in your data using advanced machine learning techniques, including Graph Neural Networks (GNN).
    
    ### What would you like to do?
    """)
    
    # Tampilkan opsi navigasi dengan kartu yang menarik
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; height: 200px;">
            <h3>Train New Model</h3>
            <p>Upload data, perform EDA, and train a new GNN model for intrusion detection</p>
            <br/>
        </div>
        """, unsafe_allow_html=True)
        train_button = st.button("Train New Model", key="train_nav")
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; height: 200px;">
            <h3>Use Existing Model</h3>
            <p>Upload data and use a pre-trained model to detect intrusions</p>
            <br/>
        </div>
        """, unsafe_allow_html=True)
        predict_button = st.button("Use Existing Model", key="predict_nav")
    
    
    if train_button:
        st.session_state['page'] = 'train'
        st.rerun()
    
    if predict_button:
        st.session_state['page'] = 'predict'
        st.rerun()
    
    # Tampilkan informasi tambahan tentang aplikasi
    st.markdown("""
    ---
    ### About this Application
    
    This Anomaly Detection Webapp enables you to analyze tabular data, perform exploratory data analysis, and detect anomalies using machine learning models.
    
    - Upload your dataset (CSV format)
    - Perform comprehensive exploratory data analysis
    - Extract relevant features for anomaly detection
    - Train or use pre-trained models to detect anomalies
    - Visualize results and download detected anomalies
    
    ### Supported Use Cases
    
    The system can be used for:
    - Fraud detection
    - Sensor fault detection
    - Unusual pattern discovery
    - And more...
    
    ### How It Works
    
    1. **Data Processing**: Your data is preprocessed and features are extracted
    2. **Graph Construction**: Data is represented as a graph for GNN analysis
    3. **Modeling**: Machine learning models analyze the data for anomalies
    4. **Detection**: Anomalous patterns are identified and visualized
    
    For detailed instructions, see the help section or the documentation.
    """)

# Define preprocessing function
def preprocess_for_anomaly_detection(df):
    """Preprocess data specifically for anomaly detection"""
    df = df.copy()
    
    # Handle missing values with robust methods
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Use median for robust imputation
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Robust scaling using StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

def preprocess_data(df):
    """Preprocess data for GNN model"""
    # Existing preprocessing code remains unchanged
    df = df.copy()
    
    # Step 1: Data Cleaning
    # Identify and remove timestamp columns
    timestamp_columns = []
    for col in df.columns:
        # Check if column name contains timestamp-related keywords
        if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'timestamp', 'datetime']):
            timestamp_columns.append(col)
        # Try to infer if it's a timestamp by checking the first few values
        elif df[col].dtype == 'object':
            sample_values = df[col].dropna().head(5).astype(str)
            # Check for common date/time patterns
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # YYYY-MM-DD HH:MM:SS
            ]
            if any(sample_values.str.contains(pattern).any() for pattern in date_patterns):
                timestamp_columns.append(col)
    
    # Remove identified timestamp columns
    if timestamp_columns:
        print(f"Removing timestamp columns: {timestamp_columns}")
        df = df.drop(columns=timestamp_columns)
    
    # Step 2: Handle IP addresses and other network identifiers
    # Identify columns that might contain IP addresses or network identifiers
    ip_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column name suggests IP or network identifier
            if any(net_keyword in col.lower() for net_keyword in ['ip', 'addr', 'address', 'src', 'dst', 'source', 'destination', 'mac', 'port']):
                ip_columns.append(col)
            # Check if values look like IP addresses
            elif df[col].dropna().head(5).astype(str).str.contains(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}').any():
                ip_columns.append(col)
    
    # Convert IP addresses to numeric features or drop them
    for col in ip_columns:
        # Option 1: Drop IP columns (simpler approach)
        df = df.drop(columns=[col])
        print(f"Dropped IP address column: {col}")
    
    # Step 3: Handle Label column - PERBAIKAN UNTUK ERROR MIX OF LABEL INPUT TYPES
    if 'Label' in df.columns:
        # Konversi semua nilai ke string terlebih dahulu untuk memastikan konsistensi
        df['Label'] = df['Label'].astype(str)
        
        # Bersihkan label string (hapus karakter khusus, konversi ke lowercase)
        df['Label'] = df['Label'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '_', x.lower()))
        
        # Tangani nilai kosong atau NaN
        df['Label'] = df['Label'].replace('', 'unknown').replace('nan', 'unknown')
        
        # Buat mapping untuk label
        unique_labels = df['Label'].unique()
        print(f"Unique labels found: {unique_labels}")
        
        # Buat mapping numerik untuk label
        label_map = {label: i for i, label in enumerate(unique_labels)}
        print(f"Label mapping: {label_map}")
        
        # Terapkan mapping ke kolom Label
        df['Label'] = df['Label'].map(label_map)
        
        # Pastikan semua nilai Label adalah numerik
        if df['Label'].isna().any():
            # Jika masih ada NaN setelah mapping, isi dengan 0 (biasanya label untuk "normal")
            print(f"Warning: {df['Label'].isna().sum()} NaN values found in Label column after mapping. Filling with 0.")
            df['Label'] = df['Label'].fillna(0).astype(int)
        else:
            df['Label'] = df['Label'].astype(int)
    
    # Step 4: Convert remaining object columns to numeric
    for col in df.columns:
        if col != 'Label' and df[col].dtype == 'object':
            try:
                # Try to convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                # If conversion fails, drop the column
                df = df.drop(columns=[col])
                print(f"Dropped column {col} due to non-numeric values")
    
    # Step 5: Handle missing values
    # Calculate percentage of missing values in each column
    missing_percentage = df.isnull().mean() * 100
    
    # Drop columns with high percentage of missing values (e.g., > 30%)
    high_missing_cols = missing_percentage[missing_percentage > 30].index.tolist()
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
        print(f"Dropped columns with >30% missing values: {high_missing_cols}")
    
    # Fill remaining missing values
    # For numeric columns, fill with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
        # Step 6: Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    # Step 7: Feature Selection
    # Select only numeric columns for feature selection
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in features:
        features.remove('Label')
    # Add debug information
    print(f"Found {len(features)} numeric features: {features}")
    
    # Check if we have any features to work with
    if len(features) == 0:
        # If no numeric features, try to convert more columns to numeric
        for col in df.columns:
            if col != 'Label':
                try:
                    # More aggressive conversion to numeric
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')
                    if not df[col].isna().all():  # Only keep if not all values are NaN
                        features.append(col)
                except:
                    pass
        
        print(f"After conversion attempt, found {len(features)} numeric features: {features}")
        
        # If still no features, raise a more informative error
        if len(features) == 0:
            # Display column types to help diagnose
            print(f"Column types in dataframe: {df.dtypes}")
            raise ValueError("No numeric features available after preprocessing. Please check your data format.")

    if len(features) > 0:
        # Step 7.1: Remove low variance features
        # Features with almost no variance don't contribute much information
        var_threshold = VarianceThreshold(threshold=0.01)
        try:
            # Handle extreme values before variance thresholding
            # Clip values to a reasonable range to prevent infinity/overflow issues
            for col in features:
                # Calculate robust statistics that aren't affected by outliers
                q1 = df[col].quantile(0.01)  # 1st percentile
                q3 = df[col].quantile(0.99)  # 99th percentile
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Clip values to the bounds
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                # Double-check for any remaining infinities or NaNs
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
            
            # Now apply variance threshold
            df_features = var_threshold.fit_transform(df[features])
            # Get the selected feature names
            selected_features = [features[i] for i in range(len(features)) 
                                if var_threshold.get_support()[i]]
            # Update the dataframe and features list
            df = pd.concat([df[selected_features], 
                           df['Label'] if 'Label' in df.columns else pd.Series()], axis=1)
            features = selected_features
            print(f"Removed low variance features. {len(selected_features)} features remaining.")
        except Exception as e:
            print(f"Variance thresholding failed: {str(e)}")
            # If variance thresholding fails, continue with original features
            selected_features = features
        
        # Step 7.2: Feature selection based on correlation with target (if Label exists)
        if 'Label' in df.columns and len(features) > 10:
            try:
                # Use mutual information for classification
                selector = SelectKBest(mutual_info_classif, k=min(20, len(features)))
                selector.fit(df[features], df['Label'])
                
                # Get selected feature names
                selected_features = [features[i] for i in range(len(features)) 
                                    if selector.get_support()[i]]
                
                # Get feature importance scores
                feature_scores = selector.scores_
                feature_importance = {features[i]: feature_scores[i] for i in range(len(features))}
                
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                print("Top features by importance:")
                for feature, score in sorted_features[:10]:
                    print(f"  {feature}: {score:.4f}")
                
                # Update the dataframe and features list
                df = pd.concat([df[selected_features], 
                               df['Label'] if 'Label' in df.columns else pd.Series()], axis=1)
                features = selected_features
                print(f"Selected top {len(selected_features)} features based on mutual information.")
            except Exception as e:
                print(f"Feature selection failed: {str(e)}")
    
    # Step 8: Normalize features
    if len(features) > 0:
        scaler = StandardScaler()
        try:
            df[features] = scaler.fit_transform(df[features])
        except Exception as e:
            # If standard scaling fails, try a more robust approach
            print(f"Standard scaling failed: {str(e)}")
            # Clip extreme values to reasonable range (e.g., 5 std devs from mean)
            for col in features:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:  # Avoid division by zero
                    df[col] = df[col].clip(mean_val - 5*std_val, mean_val + 5*std_val)
            # Try scaling again after clipping
            df[features] = scaler.fit_transform(df[features])
    
    # Step 9: Create graph connections
    edge_index = []
    for i in range(len(df) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # bidirectional
    
    # Step 10: Convert to PyTorch tensors
    if len(features) > 0:
        x = torch.tensor(df[features].values, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # If label column exists, use it
        if 'Label' in df.columns:
            y = torch.tensor(df['Label'].values, dtype=torch.long)
        else:
            y = torch.zeros(len(df), dtype=torch.long)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        return data, features
    else:
        raise ValueError("No numeric features available after preprocessing")

# Fungsi untuk melakukan EDA komprehensif
def perform_eda(df):
    st.header("Exploratory Data Analysis")
    
    # Inisialisasi processed_df di awal
    processed_df = df.copy()

    # Tab untuk berbagai jenis analisis
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "Data Overview", "Feature Analysis", "Correlation Analysis", 
        "Distribution Analysis", "Dimensionality Reduction"
    ])
    
    with eda_tab1:
        st.subheader("Data Overview")
        
        # Informasi dasar dataset
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Samples", processed_df.shape[0])
            st.metric("Number of Features", processed_df.shape[1])
        
        with col2:
            if 'Label' in processed_df.columns:
                st.metric("Number of Classes", processed_df['Label'].nunique())
                st.write("Class Distribution:")
                
                # Visualisasi distribusi kelas
                label_counts = processed_df['Label'].astype(str).value_counts().reset_index()
                label_counts.columns = ['Label', 'Count']
                
                fig = px.bar(
                    label_counts, 
                    x='Label', 
                    y='Count',
                    title="Class Distribution",
                    color='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan informasi tentang tipe data
        st.subheader("Data Types")
        
        # Hitung jumlah kolom berdasarkan tipe data
        dtype_counts = processed_df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        fig = px.pie(
            dtype_counts,
            values='Count',
            names='Data Type',
            title="Feature Data Types"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan sampel data
        st.subheader("Data Sample")
        st.dataframe(processed_df.head(10))
        
        # Tampilkan statistik deskriptif
        st.subheader("Descriptive Statistics")
        st.dataframe(processed_df.describe())
        
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            st.warning(f"Dataset contains {duplicate_rows} duplicate rows ({duplicate_rows/len(df):.2%} of data)")
            
            # Option to remove duplicates
            if st.checkbox("Remove duplicate rows"):
                df = df.drop_duplicates()
                st.success(f"Removed {duplicate_rows} duplicate rows. New shape: {df.shape}")
        else:
            st.success("No duplicate rows found in the dataset")
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.subheader("Outlier Analysis")
            
            # Pilih fitur untuk analisis outlier
            if len(numeric_cols) > 10:
                selected_features = st.multiselect(
                    "Select features for outlier analysis (max 10 recommended)",
                    options=numeric_cols,
                    default=numeric_cols[:5]
                )
            else:
                selected_features = numeric_cols

    
    with eda_tab2:
            st.subheader("Missing Data Analysis")
        
            # Check for missing values
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
                        
            # If no missing values found, show a message
            if len(missing_data) > 0:
                # Display missing data information
                missing_percent = (missing_data / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Values': missing_data.values,
                    'Percentage': missing_percent.values
                }).sort_values('Missing Values', ascending=False)
                
                st.write(f"Found {len(missing_data)} columns with missing values:")
                st.dataframe(missing_df)
                
                # Visualize missing data
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Percentage',
                    title="Percentage of Missing Values by Column",
                    color='Percentage',
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add options for handling missing data
                st.subheader("Handle Missing Data")
                
                # First, offer to convert all data to numeric
                convert_to_numeric = st.checkbox("Convert all columns to numeric before handling missing values", value=True)
                
                # Variable to track if data was processed
                data_processed = False
                processed_df = None
                
                if convert_to_numeric:
                    # Create a copy of the dataframe to avoid modifying the original
                    df_numeric = df.copy()
                    
                    # Convert all object columns to numeric
                    for col in df_numeric.columns:
                        if df_numeric[col].dtype == 'object':
                            # Try to convert directly first
                            try:
                                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                            except:
                                # If direct conversion fails, use factorize to convert categorical to numeric
                                df_numeric[col] = pd.factorize(df_numeric[col])[0]
                    
                    # Show the conversion results
                    numeric_conversion_info = pd.DataFrame({
                        'Column': df.columns,
                        'Original Type': df.dtypes,
                        'New Type': df_numeric.dtypes
                    })
                    
                    st.write("Conversion to numeric types:")
                    st.dataframe(numeric_conversion_info)
                    
                    # Update the dataframe for further processing
                    processed_df = df_numeric
                    st.success("All columns converted to numeric types")
                    data_processed = True
                
                # Options for handling missing values
                missing_handling = st.radio(
                    "Select method to handle missing values:",
                    options=["Drop rows with missing values", "Impute missing values"]
                )
                
                if missing_handling == "Drop rows with missing values":
                    # Option to drop rows with missing values
                    columns_to_check = st.multiselect(
                        "Select columns to check for missing values (empty = all columns):",
                        options=df.columns.tolist(),
                        default=[]
                    )
                    
                    if not columns_to_check:
                        columns_to_check = df.columns.tolist()
                    
                    # Preview how many rows will be dropped
                    rows_with_missing = df[columns_to_check].isnull().any(axis=1).sum()
                    st.warning(f"{rows_with_missing} rows ({rows_with_missing/len(df)*100:.2f}%) will be dropped")
                    
                    if st.button("Drop rows with missing values"):
                        # Use the processed dataframe if available, otherwise use original
                        df_to_clean = processed_df if data_processed else df.copy()
                        
                        # Drop rows with missing values
                        df_cleaned = df_to_clean.dropna(subset=columns_to_check)
                        
                        # Show results
                        st.success(f"Dropped {len(df_to_clean) - len(df_cleaned)} rows with missing values")
                        st.write(f"Original dataset shape: {df_to_clean.shape}")
                        st.write(f"Cleaned dataset shape: {df_cleaned.shape}")
                        
                        # Update the processed dataframe
                        processed_df = df_cleaned
                        data_processed = True
                
                elif missing_handling == "Impute missing values":
                    # Options for imputation
                    imputation_method = st.selectbox(
                        "Select imputation method:",
                        options=["Mean", "Median", "Mode", "Constant value", "Forward fill", "Backward fill"]
                    )
                    
                    # Columns to impute
                    columns_to_impute = st.multiselect(
                        "Select columns to impute (empty = all columns with missing values):",
                        options=missing_data.index.tolist(),
                        default=missing_data.index.tolist()
                    )
                    
                    if not columns_to_impute:
                        columns_to_impute = missing_data.index.tolist()
                    
                    # Perform imputation
                    if st.button("Impute missing values"):
                        # Use the processed dataframe if available, otherwise use original
                        df_to_impute = processed_df if data_processed else df.copy()
                        
                        # Create a copy of the dataframe
                        df_imputed = df_to_impute.copy()
                        
                        # Impute each selected column
                        for col in columns_to_impute:
                            if imputation_method == "Mean":
                                # For numeric columns, use mean
                                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                                    fill_value = df_imputed[col].mean()
                                    df_imputed[col] = df_imputed[col].fillna(fill_value)
                                    st.info(f"Column '{col}' imputed with mean: {fill_value:.4f}")
                                else:
                                    st.warning(f"Column '{col}' is not numeric, skipping mean imputation")
                            
                            elif imputation_method == "Median":
                                # For numeric columns, use median
                                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                                    fill_value = df_imputed[col].median()
                                    df_imputed[col] = df_imputed[col].fillna(fill_value)
                                    st.info(f"Column '{col}' imputed with median: {fill_value:.4f}")
                                else:
                                    st.warning(f"Column '{col}' is not numeric, skipping median imputation")
                            
                            elif imputation_method == "Mode":
                                # Use most frequent value
                                fill_value = df_imputed[col].mode()[0]
                                df_imputed[col] = df_imputed[col].fillna(fill_value)
                                st.info(f"Column '{col}' imputed with mode: {fill_value}")
                            
                            elif imputation_method == "Constant value":
                                # Use a constant value
                                fill_value = st.number_input(f"Constant value for '{col}':", value=0)
                                df_imputed[col] = df_imputed[col].fillna(fill_value)
                                st.info(f"Column '{col}' imputed with constant: {fill_value}")
                            
                            elif imputation_method == "Forward fill":
                                # Forward fill (use previous value)
                                df_imputed[col] = df_imputed[col].fillna(method='ffill')
                                if df_imputed[col].isna().any():
                                    # If still has NaN (at the beginning), use backward fill
                                    df_imputed[col] = df_imputed[col].fillna(method='bfill')
                                st.info(f"Column '{col}' imputed with forward fill")
                            
                            elif imputation_method == "Backward fill":
                                # Backward fill (use next value)
                                df_imputed[col] = df_imputed[col].fillna(method='bfill')
                                if df_imputed[col].isna().any():
                                    # If still has NaN (at the end), use forward fill
                                    df_imputed[col] = df_imputed[col].fillna(method='ffill')
                                st.info(f"Column '{col}' imputed with backward fill")
                        
                        # Check if any missing values remain
                        remaining_missing = df_imputed.isnull().sum().sum()
                        if remaining_missing > 0:
                            st.warning(f"{remaining_missing} missing values remain after imputation")
                            # Fill any remaining missing values with 0
                            df_imputed = df_imputed.fillna(0)
                            st.info("Remaining missing values filled with 0")
                        
                        # Show results
                        st.success("Missing values imputed successfully")
                        
                        # Update the processed dataframe
                        processed_df = df_imputed
                        data_processed = True
              
            # Add column removal functionality
            st.subheader("Remove Unnecessary Columns")
            
            # Display all available columns
            all_columns = processed_df.columns.tolist()
            
            # Allow user to select columns to drop
            columns_to_drop = st.multiselect(
                "Select columns to remove from the dataset:",
                options=all_columns,
                default=[]
            )
            
            # Show preview of selected columns
            if columns_to_drop:
                st.write(f"You've selected {len(columns_to_drop)} columns to drop:")
                for col in columns_to_drop:
                    st.write(f"- {col}")
                
                # Preview the dataset after dropping columns
                preview_df = processed_df.drop(columns=columns_to_drop)
                st.write(f"Dataset shape after dropping columns: {preview_df.shape}")
                
                # Button to confirm dropping columns
                if st.button("Confirm and Drop Selected Columns"):
                    # Create a copy to avoid modifying the original dataframe
                    df = processed_df.drop(columns=columns_to_drop)
                    st.success(f"Successfully dropped {len(columns_to_drop)} columns")
                    st.write("Preview of the processed dataset:")
                    st.dataframe(processed_df.head())
                    st.success("Dataset updated successfully!")
                    # Update the processed dataframe
                    processed_df = df
                    data_processed = True
            else:
                st.success("No missing values found in the dataset")
    
    with eda_tab3:
        st.subheader("Correlation Analysis")
        
        # Use the original dataframe
        analysis_df = processed_df.copy()
        
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Hitung matriks korelasi
            corr_matrix = analysis_df[numeric_cols].corr()
            
            # Visualisasi matriks korelasi dengan heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                corr_matrix, 
                mask=mask, 
                cmap=cmap, 
                vmax=1, 
                vmin=-1, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5},
                annot=False
            )
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig)
            
            # Tampilkan fitur dengan korelasi tinggi
            st.subheader("High Correlation Pairs")
            
            # Dapatkan pasangan fitur dengan korelasi tinggi
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.7:  # Threshold untuk korelasi tinggi
                        corr_pairs.append({
                            'Feature 1': numeric_cols[i],
                            'Feature 2': numeric_cols[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if corr_pairs:
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df)
                
                # Visualisasi scatter plot untuk pasangan dengan korelasi tertinggi
                if len(corr_pairs) > 0:
                    top_pair = corr_pairs[0]
                    fig = px.scatter(
                        analysis_df, 
                        x=top_pair['Feature 1'], 
                        y=top_pair['Feature 2'],
                        title=f"Scatter Plot: {top_pair['Feature 1']} vs {top_pair['Feature 2']} (Correlation: {top_pair['Correlation']:.3f})",
                        color='Label' if 'Label' in analysis_df.columns else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature pairs with high correlation (>0.7) found")
            
            # Korelasi dengan target (jika ada)
            if 'Label' in analysis_df.columns and analysis_df['Label'].dtype != 'object':
                st.subheader("Feature Correlation with Target")
                
                # Hitung korelasi dengan target
                target_corr = []
                for col in numeric_cols:
                    if col != 'Label':
                        corr, _ = pearsonr(analysis_df[col], analysis_df['Label'])
                        target_corr.append({
                            'Feature': col,
                            'Correlation': corr,
                            'Abs Correlation': abs(corr)
                        })
                
                target_corr_df = pd.DataFrame(target_corr).sort_values('Abs Correlation', ascending=False)
                
                # Visualisasi korelasi dengan target
                fig = px.bar(
                    target_corr_df.head(20),
                    x='Feature',
                    y='Correlation',
                    title="Top 20 Features by Correlation with Target",
                    color='Correlation',
                    color_continuous_scale=px.colors.diverging.RdBu_r
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric features for correlation analysis")
    
    with eda_tab4:
        st.subheader("Distribution Analysis")
        
        # Analisis distribusi fitur berdasarkan kelas (jika ada)
        if 'Label' in processed_df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            if len(numeric_cols) > 0:
                # Pilih fitur untuk analisis distribusi
                selected_feature = st.selectbox(
                    "Select feature for distribution analysis",
                    options=numeric_cols
                )
                
                # Visualisasi distribusi berdasarkan kelas
                fig = px.histogram(
                    df, 
                    x=selected_feature,
                    color='Label',
                    marginal="box",
                    barmode="overlay",
                    title=f"Distribution of {selected_feature} by Class"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Violin plot untuk perbandingan distribusi
                fig = px.violin(
                    df, 
                    y=selected_feature, 
                    x='Label',
                    box=True,
                    title=f"Violin Plot: {selected_feature} by Class"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Analisis distribusi umum
        st.subheader("General Distribution Analysis")
        
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            # Pilih beberapa fitur untuk analisis
            if len(numeric_cols) > 5:
                selected_features = st.multiselect(
                    "Select features for distribution comparison (max 5 recommended)",
                    options=numeric_cols,
                    default=numeric_cols[:3]
                )
            else:
                selected_features = numeric_cols
            
            if selected_features:
                # Normalisasi data untuk perbandingan
                normalized_df = df[selected_features].copy()
                for feature in selected_features:
                    # Pastikan kolom adalah numerik
                    if pd.api.types.is_numeric_dtype(normalized_df[feature]):
                        # Tangani kasus nilai min dan max sama
                        min_val = normalized_df[feature].min()
                        max_val = normalized_df[feature].max()
                        if min_val == max_val:
                            normalized_df[feature] = 0  # atau 1, tergantung kebutuhan
                        else:
                            normalized_df[feature] = (normalized_df[feature] - min_val) / (max_val - min_val)
                    else:
                        st.warning(f"Kolom {feature} bukan numerik dan akan dilewati")
                        normalized_df[feature] = 0
                
                # Reshape data untuk visualisasi
                melted_df = pd.melt(normalized_df, value_vars=selected_features, var_name='Feature', value_name='Normalized Value')
                
                # Visualisasi perbandingan distribusi
                fig = px.violin(
                    melted_df, 
                    y='Normalized Value', 
                    x='Feature',
                    box=True,
                    title="Normalized Feature Distributions"
                )
                st.plotly_chart(fig, use_container_width=True)
    
        with eda_tab5:
            st.subheader("Dimensionality Reduction")
            
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
            if len(numeric_cols) > 2:
                # Pilih metode dimensionality reduction
                dim_reduction_method = st.radio(
                    "Select dimensionality reduction method",
                    options=["PCA", "t-SNE"]
                )
                
                # Persiapkan data
                X = df[numeric_cols].copy()
                
                # Handle infinite values and extreme outliers
                # Replace infinities with NaN first
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # For each column, handle extreme values using percentile-based clipping
                for col in X.columns:
                    # Check if column has any non-NaN values and is numeric
                    if X[col].notna().any() and pd.api.types.is_numeric_dtype(X[col]):
                        try:
                            # Calculate robust statistics that aren't affected by outliers
                            q1 = X[col].quantile(0.01)  # 1st percentile
                            q3 = X[col].quantile(0.99)  # 99th percentile
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr
                            upper_bound = q3 + 3 * iqr
                            
                            # Clip values to the bounds
                            X[col] = X[col].clip(lower_bound, upper_bound)
                        except TypeError as e:
                            st.warning(f"Could not process column '{col}' due to data type issues. Converting to numeric.")
                            # Try to convert to numeric, replacing errors with NaN
                            X[col] = pd.to_numeric(X[col], errors='coerce')
                            
                            # If conversion successful, try again with outlier removal
                            if pd.api.types.is_numeric_dtype(X[col]) and X[col].notna().any():
                                q1 = X[col].quantile(0.01)
                                q3 = X[col].quantile(0.99)
                                iqr = q3 - q1
                                lower_bound = q1 - 3 * iqr
                                upper_bound = q3 + 3 * iqr
                                X[col] = X[col].clip(lower_bound, upper_bound)
                    
                    # Fill remaining NaNs with median
                    if X[col].isna().any():
                        if X[col].notna().any():
                            try:
                                # Try to convert to numeric first
                                X[col] = pd.to_numeric(X[col], errors='coerce')
                                # Then calculate median on numeric values
                                median_val = X[col].median()
                                # Fill NaN values with the median
                                X[col] = X[col].fillna(median_val)
                            except:
                                # If median calculation fails, use 0
                                X[col] = X[col].fillna(0)
                        else:
                            X[col] = X[col].fillna(0)
                
                # Double-check for any remaining infinities or NaNs
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(0)  # Fill any remaining NaNs with 0
                
                
                if dim_reduction_method == "PCA":
                    # Lakukan PCA
                    try:
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X)
                        
                        # Buat dataframe hasil PCA
                        pca_df = pd.DataFrame({
                            'PCA1': pca_result[:, 0],
                            'PCA2': pca_result[:, 1]
                        })
                        
                        # Tambahkan label jika ada
                        if 'Label' in df.columns:
                            pca_df['Label'] = df['Label'].values
                        
                        # Visualisasi hasil PCA
                        fig = px.scatter(
                            pca_df, 
                            x='PCA1', 
                            y='PCA2',
                            color='Label' if 'Label' in pca_df.columns else None,
                            title="PCA Visualization"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan explained variance
                        explained_variance = pca.explained_variance_ratio_
                        st.write(f"Explained variance ratio: {explained_variance[0]:.4f}, {explained_variance[1]:.4f}")
                        st.write(f"Total explained variance: {sum(explained_variance):.4f}")
                        
                        # Visualisasi explained variance
                        fig = px.bar(
                            x=['PC1', 'PC2'],
                            y=explained_variance,
                            title="Explained Variance by Principal Component"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance dalam PCA
                        st.subheader("Feature Importance in PCA")
                        
                        # Dapatkan loadings
                        loadings = pca.components_.T
                        
                        # Buat dataframe untuk loadings
                        loadings_df = pd.DataFrame({
                            'Feature': numeric_cols,
                            'PC1': loadings[:, 0],
                            'PC2': loadings[:, 1],
                            'Magnitude': np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
                        }).sort_values('Magnitude', ascending=False)
                        
                        st.dataframe(loadings_df)
                        
                        # Visualisasi loadings
                        fig = px.scatter(
                            loadings_df, 
                            x='PC1', 
                            y='PC2',
                            text='Feature',
                            title="PCA Loadings"
                        )
                        fig.update_traces(textposition='top center')
                        fig.update_layout(
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            xaxis_zeroline=True,
                            yaxis_zeroline=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"PCA failed: {str(e)}")
                        st.info("Try removing more outliers or using a different dimensionality reduction method.")
                
                elif dim_reduction_method == "t-SNE":
                    try:
                        # Sample data if it's too large (t-SNE is computationally expensive)
                        max_samples = 5000
                        if len(X) > max_samples:
                            st.info(f"Dataset is large. Sampling {max_samples} points for t-SNE visualization.")
                            # Use stratified sampling if Label exists
                            if 'Label' in df.columns:
                                # Ensure all classes are represented in the sample
                                sample_indices = []
                                labels = df['Label'].unique()
                                
                                # Calculate samples per class
                                samples_per_class = max(1, max_samples // len(labels))
                                
                                for label in labels:
                                    label_indices = df[df['Label'] == label].index.tolist()
                                    # Take either all indices or the calculated number, whichever is smaller
                                    n_samples = min(len(label_indices), samples_per_class)
                                    if n_samples > 0:
                                        sampled = np.random.choice(label_indices, size=n_samples, replace=False)
                                        sample_indices.extend(sampled)
                                
                                # If we need more samples to reach max_samples, add random samples
                                if len(sample_indices) < max_samples:
                                    remaining = max_samples - len(sample_indices)
                                    # Get indices not already in sample_indices
                                    remaining_indices = [i for i in range(len(df)) if i not in sample_indices]
                                    if remaining_indices:
                                        additional = np.random.choice(
                                            remaining_indices, 
                                            size=min(remaining, len(remaining_indices)), 
                                            replace=False
                                        )
                                        sample_indices.extend(additional)
                                
                                X_sample = X.iloc[sample_indices]
                                labels_sample = df.iloc[sample_indices]['Label'] if 'Label' in df.columns else None
                            else:
                                # Random sampling if no Label
                                sample_indices = np.random.choice(len(X), size=min(max_samples, len(X)), replace=False)
                                X_sample = X.iloc[sample_indices]
                                labels_sample = None
                        else:
                            X_sample = X
                            labels_sample = df['Label'] if 'Label' in df.columns else None
                        
                        # Standardize the data for t-SNE
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_sample)
                        
                        # Check for any remaining infinities or NaNs after scaling
                        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Run t-SNE with progress information
                        perplexity = min(30, len(X_scaled) - 1)  # Adjust perplexity based on sample size
                        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
                        
                        with st.spinner("Running t-SNE dimensionality reduction..."):
                            tsne_result = tsne.fit_transform(X_scaled)
                        
                        # Create dataframe for visualization
                        tsne_df = pd.DataFrame({
                            'TSNE1': tsne_result[:, 0],
                            'TSNE2': tsne_result[:, 1]
                        })
                        
                        # Add labels if available
                        if labels_sample is not None:
                            tsne_df['Label'] = labels_sample.values
                        
                        # Visualize t-SNE results
                        fig = px.scatter(
                            tsne_df, 
                            x='TSNE1', 
                            y='TSNE2',
                            color='Label' if 'Label' in tsne_df.columns else None,
                            title="t-SNE Visualization"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add information about t-SNE
                        st.info("""
                        t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique 
                        that visualizes high-dimensional data by giving each datapoint a location in a 2D or 3D map. 
                        It's particularly good at revealing clusters of similar data points.
                        
                        Note: Unlike PCA, t-SNE doesn't preserve global structure, so distances between separated 
                        clusters may not be meaningful.
                        """)
                        
                    except Exception as e:
                        st.error(f"t-SNE failed: {str(e)}")
                        st.info("""
                        Try these solutions:
                        1. Reduce the number of features before applying t-SNE
                        2. Remove outliers more aggressively
                        3. Use PCA instead, which is more robust to outliers
                        """)
            else:
                st.warning("Not enough numeric features for dimensionality reduction. Need at least 3 numeric columns.")

# Anomaly detection evaluation
def evaluate_anomaly_detection(y_true, y_pred, scores):
    """Evaluate anomaly detection performance"""
    
    # If no ground truth, use unsupervised metrics
    if y_true is None or len(np.unique(y_true)) == 1:
        return {
            'total_anomalies': np.sum(y_pred),
            'anomaly_rate': np.mean(y_pred),
            'score_distribution': {
                'min': np.min(scores),
                'max': np.max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        }
    
    # Supervised evaluation
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, scores),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

# Anomaly detection pipeline
def run_anomaly_detection_pipeline(data, method='isolation_forest', contamination=0.1):
    """Run complete anomaly detection pipeline"""
    
    # Method selection
    if method == 'isolation_forest':
        model = IsolationForest(contamination=contamination, random_state=42)
    elif method == 'lof':
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    elif method == 'one_class_svm':
        model = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
    
    # Fit and predict
    if method != 'lof':
        model.fit(data)
        anomaly_scores = model.decision_function(data)
        predictions = model.predict(data)
    else:
        predictions = model.fit_predict(data)
        anomaly_scores = model.negative_outlier_factor_
    
    # Convert to binary labels
    anomaly_labels = np.where(predictions == -1, 1, 0)
    
    return {
        'anomaly_labels': anomaly_labels,
        'anomaly_scores': anomaly_scores,
        'model': model
    }

# Enhanced visualization functions
def visualize_anomalies(df, anomaly_labels, anomaly_scores):
    """Create comprehensive anomaly visualizations"""
    
    # Scatter plot with anomalies highlighted
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                     color=anomaly_labels.astype(str),
                     title='Anomaly Detection Results',
                     labels={'color': 'Anomaly'})
    
    # Anomaly score distribution
    score_fig = px.histogram(anomaly_scores, nbins=50,
                           title='Anomaly Score Distribution')
    
    # 3D visualization if enough features
    if len(df.columns) >= 3:
        fig_3d = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2],
                              color=anomaly_labels.astype(str),
                              title='3D Anomaly Visualization')
        return fig, score_fig, fig_3d
    
    return fig, score_fig

# Fungsi untuk ekstraksi fitur
def extract_features(df, features):
    st.subheader("Feature Extraction and Selection")
    
    # Create tabs for different feature operations
    feature_tab1, feature_tab2, feature_tab3 = st.tabs(["Feature Importance", "Feature Selection", "Feature Engineering"])
    
    
    with feature_tab1:
        st.subheader("Feature Importance Analysis")

        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')

        
        # Allow manual selection of target column
        if 'Label' in all_columns:
            default_target = 'Label'
        else:
            default_target = all_columns[0] if all_columns else None
                
        target_col = st.selectbox(
            "Select target column for feature importance analysis:",
            options=all_columns,
            index=all_columns.index(default_target) if default_target in all_columns else 0
        )
        
        # Check if target column exists
        if target_col:
            has_target = True
            
            # Allow manual selection of features for analysis
            available_features = [col for col in numeric_cols if col != target_col]
            
            if available_features:
                # Option to use all numeric features or select specific ones
                use_all_features = st.checkbox("Use all numeric features", value=True)
                
                if use_all_features:
                    selected_features = available_features
                else:
                    selected_features = st.multiselect(
                        "Select numeric features for importance analysis:",
                        options=available_features,
                        default=available_features[:min(10, len(available_features))]
                    )
                
                # Check if we have features to analyze
                if selected_features:
                    # Select feature importance method
                    importance_method = st.selectbox(
                        "Select feature importance method:",
                        options=["Mutual Information", "Random Forest", "Permutation Importance"]
                    )
                    
                    try:
                        # Ensure target column is properly formatted for analysis
                        target_values = df[target_col].copy()
                        
                        # Convert target to numeric if it's not already
                        if df[target_col].dtype == 'object':
                            st.info(f"Converting categorical target '{target_col}' to numeric for analysis")
                            # Create a mapping for categorical values
                            unique_values = df[target_col].unique()
                            target_map = {val: i for i, val in enumerate(unique_values)}
                            target_values = df[target_col].map(target_map)
                            
                            # Display the mapping for user reference
                            mapping_df = pd.DataFrame({
                                'Original Value': list(target_map.keys()),
                                'Numeric Value': list(target_map.values())
                            })
                            show_mapping = st.checkbox("Show category to numeric mapping")
                            if show_mapping:
                                st.dataframe(mapping_df)
                        
                        # Prepare feature data - handle missing and infinite values
                        X = df[selected_features].copy()
                        X = X.replace([np.inf, -np.inf], np.nan)
                        
                        # Fill missing values with median for each column
                        for col in X.columns:
                            if X[col].isna().any():
                                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
                        
                        # Calculate feature importance based on selected method
                        if importance_method == "Mutual Information":
                            # Determine if classification or regression task
                            unique_labels = len(np.unique(target_values))
                            is_classification = unique_labels < 10  # Heuristic
                            
                            if is_classification:
                                # For classification tasks
                                from sklearn.feature_selection import mutual_info_classif
                                importance_scores = mutual_info_classif(X, target_values)
                                method_name = "Mutual Information (Classification)"
                            else:
                                # For regression tasks
                                from sklearn.feature_selection import mutual_info_regression
                                importance_scores = mutual_info_regression(X, target_values)
                                method_name = "Mutual Information (Regression)"
                                
                        elif importance_method == "Random Forest":
                            # Determine if classification or regression task
                            unique_labels = len(np.unique(target_values))
                            is_classification = unique_labels < 10  # Heuristic
                            
                            if is_classification:
                                # For classification tasks
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                                model.fit(X, target_values)
                                importance_scores = model.feature_importances_
                                method_name = "Random Forest Importance (Classification)"
                            else:
                                # For regression tasks
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model.fit(X, target_values)
                                importance_scores = model.feature_importances_
                                method_name = "Random Forest Importance (Regression)"
                                
                        elif importance_method == "Permutation Importance":
                            # Determine if classification or regression task
                            unique_labels = len(np.unique(target_values))
                            is_classification = unique_labels < 10  # Heuristic
                            
                            from sklearn.inspection import permutation_importance
                            
                            if is_classification:
                                # For classification tasks
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=50, random_state=42)
                                model.fit(X, target_values)
                                
                                # Calculate permutation importance
                                result = permutation_importance(model, X, target_values, n_repeats=10, random_state=42)
                                importance_scores = result.importances_mean
                                method_name = "Permutation Importance (Classification)"
                            else:
                                # For regression tasks
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=50, random_state=42)
                                model.fit(X, target_values)
                                
                                # Calculate permutation importance
                                result = permutation_importance(model, X, target_values, n_repeats=10, random_state=42)
                                importance_scores = result.importances_mean
                                method_name = "Permutation Importance (Regression)"
                        
                        # Create dataframe for feature scores
                        feature_scores = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': importance_scores
                        }).sort_values('Importance', ascending=False)
                        
                        # Visualize feature importance
                        fig = px.bar(
                            feature_scores.head(20),
                            x='Feature',
                            y='Importance',
                            title=f"Top 20 Features by {method_name} (Target: {target_col})",
                            color='Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display feature importance table
                        st.dataframe(feature_scores)
                        
                        # Option to add top features to the dataset
                        if st.checkbox("Add feature importance as a column to the dataset"):
                            top_n = st.slider("Number of top features to include", 1, len(selected_features), 
                                             min(5, len(selected_features)))
                            top_features = feature_scores.head(top_n)['Feature'].tolist()
                            st.success(f"Selected top {top_n} features: {', '.join(top_features)}")
                            return df, top_features
                        
                    except Exception as e:
                        st.error(f"Error calculating feature importance: {str(e)}")
                        st.info("Tip: Make sure your target column and selected features are compatible for analysis")
                        st.exception(e)  # Show detailed error for debugging
                else:
                    st.warning("Please select at least one feature for importance analysis")
            else:
                st.warning("No numeric features available for analysis")
        else:
            st.error("No columns available to use as target")
            has_target = False
        
    with feature_tab2:
        st.subheader("Feature Selection")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = [col for col in selected_features if col in numeric_cols]

        if 'Label' in selected_features:
            selected_features.remove('Label')

        # Extract features and labels
        X = df[selected_features].values
        y = df['Label'].values
            
        if len(numeric_cols) > 0:
                # Pilih metode seleksi fitur
                selection_method = st.radio(
                    "Select feature selection method",
                    options=["Variance Threshold", "SelectKBest (ANOVA F-value)", "SelectKBest (Mutual Information)", 
                             "HistGradientBoosting Importance"]
                )
                
                if selection_method == "Variance Threshold":
                    # Pilih threshold
                    variance_threshold = st.slider(
                        "Variance threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.01,
                        step=0.01
                    )
                    
                    # Lakukan seleksi fitur
                    try:
                        selector = VarianceThreshold(threshold=variance_threshold)
                        X_selected = selector.fit_transform(df[numeric_cols])
                        
                        # Dapatkan fitur yang dipilih
                        selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                                            if selector.get_support()[i]]
                        
                        # Tampilkan hasil
                        st.success(f"Selected {len(selected_features)} features out of {len(numeric_cols)}")
                        st.write("Selected features:")
                        st.write(selected_features)
                        
                        # Tampilkan variance untuk setiap fitur
                        variance_df = pd.DataFrame({
                            'Feature': numeric_cols,
                            'Variance': selector.variances_,
                            'Selected': selector.get_support()
                        }).sort_values('Variance', ascending=False)
                        
                        st.dataframe(variance_df)
                        
                        # Visualisasi variance
                        fig = px.bar(
                            variance_df,
                            x='Feature',
                            y='Variance',
                            color='Selected',
                            title="Feature Variance",
                            color_discrete_map={True: 'green', False: 'red'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in variance thresholding: {str(e)}")
                
                elif selection_method == "SelectKBest (ANOVA F-value)":
                    if 'Label' in df.columns:
                        # Pilih jumlah fitur
                        k_features = st.slider(
                            "Number of features to select",
                            min_value=1,
                            max_value=min(20, len(numeric_cols)),
                            value=min(10, len(numeric_cols))
                        )
                        
                        # Lakukan seleksi fitur
                        try:
                            selector = SelectKBest(f_classif, k=k_features)
                            X_selected = selector.fit_transform(df[numeric_cols], df['Label'])
                            
                            # Dapatkan fitur yang dipilih
                            selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                                                if selector.get_support()[i]]
                            
                            # Tampilkan hasil
                            st.success(f"Selected {len(selected_features)} features using ANOVA F-value")
                            
                            # Buat dataframe untuk skor fitur
                            feature_scores = pd.DataFrame({
                                'Feature': numeric_cols,
                                'Score': selector.scores_,
                                'P-value': selector.pvalues_,
                                'Selected': selector.get_support()
                            }).sort_values('Score', ascending=False)
                            
                            st.dataframe(feature_scores)
                            
                            # Visualisasi skor fitur
                            fig = px.bar(
                                feature_scores.head(20),
                                x='Feature',
                                y='Score',
                                color='Selected',
                                title="Feature Scores (ANOVA F-value)",
                                color_discrete_map={True: 'green', False: 'gray'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Visualisasi p-values
                            fig = px.bar(
                                feature_scores.head(20),
                                x='Feature',
                                y='P-value',
                                color='Selected',
                                title="Feature P-values (ANOVA F-value)",
                                color_discrete_map={True: 'green', False: 'gray'},
                                log_y=True
                            )
                            fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="p=0.05")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in ANOVA F-value feature selection: {str(e)}")
                    else:
                        st.warning("Label column required for ANOVA F-value feature selection")
                
                elif selection_method == "SelectKBest (Mutual Information)":
                    if 'Label' in df.columns:
                        # Pilih jumlah fitur
                        k_features = st.slider(
                            "Number of features to select",
                            min_value=1,
                            max_value=min(20, len(numeric_cols)),
                            value=min(10, len(numeric_cols))
                        )
                        
                        # Lakukan seleksi fitur
                        try:
                            selector = SelectKBest(mutual_info_classif, k=k_features)
                            X_selected = selector.fit_transform(df[numeric_cols], df['Label'])
                            
                            # Dapatkan fitur yang dipilih
                            selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                                                if selector.get_support()[i]]
                            
                            # Tampilkan hasil
                            st.success(f"Selected {len(selected_features)} features using Mutual Information")
                            
                            # Buat dataframe untuk skor fitur
                            feature_scores = pd.DataFrame({
                                'Feature': numeric_cols,
                                'Score': selector.scores_,
                                'Selected': selector.get_support()
                            }).sort_values('Score', ascending=False)
                            
                            st.dataframe(feature_scores)
                            
                            # Visualisasi skor fitur
                            fig = px.bar(
                                feature_scores.head(20),
                                x='Feature',
                                y='Score',
                                color='Selected',
                                title="Feature Scores (Mutual Information)",
                                color_discrete_map={True: 'green', False: 'gray'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in Mutual Information feature selection: {str(e)}")
                    else:
                        st.warning("Label column required for Mutual Information feature selection")
   

                elif selection_method == "HistGradientBoosting Importance":
                    if 'Label' in df.columns:
                        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
                        
                        # Check if we need classification or regression
                        unique_labels = df['Label'].nunique()
                        is_classification = unique_labels < 10  # Heuristic: if fewer than 10 unique values, assume classification
                        
                        # Let user override the detection
                        task_type = st.radio(
                            "Task type",
                            options=["Classification", "Regression"],
                            index=0 if is_classification else 1
                        )
                        
                        # Number of features to select
                        k_features = st.slider(
                            "Number of features to select",
                            min_value=1,
                            max_value=min(20, len(numeric_cols)),
                            value=min(10, len(numeric_cols))
                        )
                        
                        # Model parameters
                        learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
                        max_iter = st.slider("Maximum iterations", 10, 500, 100, 10)
                        max_depth = st.slider("Maximum tree depth", 1, 20, 10, 1)
                        
                        try:
                            # Prepare data
                            X = df[numeric_cols].copy()
                            y = df['Label'].copy()
                            
                            # Handle non-numeric target for regression
                            if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                                st.warning("Converting non-numeric target to numeric for regression")
                                # Map categorical values to numbers
                                y = pd.factorize(y)[0]
                            
                            # Create and train the model
                            with st.spinner("Training HistGradientBoosting model..."):
                                if task_type == "Classification":
                                    model = HistGradientBoostingClassifier(
                                        learning_rate=learning_rate,
                                        max_iter=max_iter,
                                        max_depth=max_depth,
                                        random_state=42
                                    )
                                else:  # Regression
                                    model = HistGradientBoostingRegressor(
                                        learning_rate=learning_rate,
                                        max_iter=max_iter,
                                        max_depth=max_depth,
                                        random_state=42
                                    )
                                
                                # Replace infinities and NaNs
                                X = X.replace([np.inf, -np.inf], np.nan)
                                
                                # Handle NaNs with safer median calculation
                                for col in X.columns:
                                    if X[col].isna().any():
                                        try:
                                            # Try to convert to numeric first
                                            X[col] = pd.to_numeric(X[col], errors='coerce')
                                            # Calculate median on numeric values
                                            median_val = X[col].median()
                                            # Fill NaN values with the median
                                            X[col] = X[col].fillna(median_val)
                                        except:
                                            # If median calculation fails, use 0
                                            X[col] = X[col].fillna(0)
                                
                                # Fit the model
                                model.fit(X, y)
                            
                            # Get feature importances
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                
                                # Create dataframe for feature importances
                                importance_df = pd.DataFrame({
                                    'Feature': numeric_cols,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                # Select top k features
                                selected_features = importance_df.head(k_features)['Feature'].tolist()
                                
                                # Display results
                                st.success(f"Selected {len(selected_features)} features using HistGradientBoosting importance")
                                
                                # Add a 'Selected' column for visualization
                                importance_df['Selected'] = importance_df['Feature'].isin(selected_features)
                                
                                # Show feature importance table
                                st.dataframe(importance_df)
                                
                                # Visualize feature importances
                                fig = px.bar(
                                    importance_df.head(20),
                                    x='Feature',
                                    y='Importance',
                                    color='Selected',
                                    title=f"Feature Importance (HistGradientBoosting {task_type})",
                                    color_discrete_map={True: 'green', False: 'gray'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Option to use selected features
                                if st.checkbox("Use these selected features for further analysis"):
                                    st.success(f"Using top {k_features} features: {', '.join(selected_features)}")
                                    return df, selected_features
                            else:
                                st.error("Model doesn't have feature_importances_ attribute")
                        except Exception as e:
                            st.error(f"Error in HistGradientBoosting feature selection: {str(e)}")
                            st.info("Try adjusting the parameters or preprocessing the data")
                    else:
                        st.warning("Label column required for HistGradientBoosting feature selection")
                else:
                    st.warning("No numeric features found for feature selection")    

    with feature_tab3:
        st.subheader("Feature Engineering")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
                numeric_cols.remove('Label')
            
        if len(numeric_cols) > 0:
                # Pilih metode feature engineering
                engineering_method = st.radio(
                    "Select feature engineering method",
                    options=["Polynomial Features", "PCA Transformation", "Custom Feature Combinations"]
                )
                
                if engineering_method == "Polynomial Features":
                    # Pilih derajat polinomial
                    poly_degree = st.slider(
                        "Polynomial degree",
                        min_value=2,
                        max_value=3,
                        value=2
                    )
                    
                    # Pilih fitur untuk transformasi polinomial
                    if len(numeric_cols) > 5:
                        selected_features = st.multiselect(
                            "Select features for polynomial transformation (max 5 recommended)",
                            options=numeric_cols,
                            default=numeric_cols[:2]
                        )
                    else:
                        selected_features = numeric_cols
                    
                    if selected_features:
                        # Lakukan transformasi polinomial
                        try:
                            from sklearn.preprocessing import PolynomialFeatures
                            
                            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                            poly_features = poly.fit_transform(df[selected_features])
                            
                            # Dapatkan nama fitur baru
                            feature_names = poly.get_feature_names_out(selected_features)
                            
                            # Buat dataframe dengan fitur polinomial
                            poly_df = pd.DataFrame(poly_features, columns=feature_names)
                            
                            # Tampilkan hasil
                            st.success(f"Created {poly_df.shape[1]} polynomial features from {len(selected_features)} original features")
                            
                            # Tampilkan sampel data
                            st.write("Sample of polynomial features:")
                            st.dataframe(poly_df.head())
                            
                            # Visualisasi distribusi fitur polinomial
                            if st.checkbox("Visualize polynomial feature distributions"):
                                # Pilih fitur untuk visualisasi
                                poly_viz_feature = st.selectbox(
                                    "Select polynomial feature to visualize",
                                    options=feature_names
                                )
                                
                                # Histogram
                                fig = px.histogram(
                                    poly_df,
                                    x=poly_viz_feature,
                                    title=f"Distribution of {poly_viz_feature}",
                                    marginal="box"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in polynomial feature generation: {str(e)}")
                
                elif engineering_method == "PCA Transformation":
                    # Pilih jumlah komponen
                    n_components = st.slider(
                        "Number of PCA components",
                        min_value=2,
                        max_value=min(10, len(numeric_cols)),
                        value=min(5, len(numeric_cols))
                    )
                    
                    # Lakukan transformasi PCA
                    try:
                        # Standardisasi data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[numeric_cols])
                        
                        # Lakukan PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Buat dataframe dengan komponen PCA
                        pca_df = pd.DataFrame(
                            pca_result,
                            columns=[f"PC{i+1}" for i in range(n_components)]
                        )
                        
                        # Tampilkan hasil
                        st.success(f"Created {n_components} PCA components from {len(numeric_cols)} original features")
                        
                        # Tampilkan explained variance
                        explained_variance = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        # Visualisasi explained variance
                        variance_df = pd.DataFrame({
                            'Component': [f"PC{i+1}" for i in range(n_components)],
                            'Explained Variance': explained_variance,
                            'Cumulative Variance': cumulative_variance
                        })
                        
                        # Tampilkan tabel
                        st.write("Explained variance by component:")
                        st.dataframe(variance_df)
                        
                        # Visualisasi explained variance
                        fig = px.bar(
                            variance_df,
                            x='Component',
                            y='Explained Variance',
                            title="Explained Variance by PCA Component"
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=variance_df['Component'],
                                y=variance_df['Cumulative Variance'],
                                mode='lines+markers',
                                name='Cumulative Variance',
                                yaxis='y2'
                            )
                        )
                        fig.update_layout(
                            yaxis2=dict(
                                title='Cumulative Variance',
                                overlaying='y',
                                side='right',
                                range=[0, 1]
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan sampel data
                        st.write("Sample of PCA components:")
                        st.dataframe(pca_df.head())
                        
                        # Visualisasi loadings
                        loadings = pca.components_
                        loadings_df = pd.DataFrame(
                            loadings.T,
                            columns=[f"PC{i+1}" for i in range(n_components)],
                            index=numeric_cols
                        )
                        
                        st.write("PCA Loadings (feature contributions to each component):")
                        st.dataframe(loadings_df)
                        
                        # Heatmap loadings
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.heatmap(loadings_df, annot=True, cmap='coolwarm', ax=ax)
                        plt.title('PCA Loadings Heatmap')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error in PCA transformation: {str(e)}")
                
                elif engineering_method == "Custom Feature Combinations":
                    st.write("Create custom feature combinations by applying operations to existing features")
                    
                    # Pilih fitur untuk kombinasi
                    if len(numeric_cols) > 0:
                        feature1 = st.selectbox(
                            "Select first feature",
                            options=numeric_cols
                        )
                        
                        feature2 = st.selectbox(
                            "Select second feature",
                            options=[col for col in numeric_cols if col != feature1],
                            index=0 if len(numeric_cols) > 1 else None
                        )
                        
                        # Pilih operasi
                        operation = st.radio(
                            "Select operation",
                            options=["Sum", "Difference", "Product", "Ratio", "Mean"]
                        )
                        
                        if feature1 and feature2:
                            try:
                                # Lakukan operasi
                                if operation == "Sum":
                                    result = df[feature1] + df[feature2]
                                    new_feature_name = f"{feature1}_plus_{feature2}"
                                elif operation == "Difference":
                                    result = df[feature1] - df[feature2]
                                    new_feature_name = f"{feature1}_minus_{feature2}"
                                elif operation == "Product":
                                    result = df[feature1] * df[feature2]
                                    new_feature_name = f"{feature1}_times_{feature2}"
                                elif operation == "Ratio":
                                    # Handle division by zero
                                    result = df[feature1] / df[feature2].replace(0, np.nan)
                                    new_feature_name = f"{feature1}_div_{feature2}"
                                elif operation == "Mean":
                                    result = (df[feature1] + df[feature2]) / 2
                                    new_feature_name = f"mean_{feature1}_{feature2}"
                                
                                # Tampilkan hasil
                                st.success(f"Created new feature: {new_feature_name}")
                                
                                # Statistik fitur baru
                                stats = pd.DataFrame({
                                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                    'Value': [
                                        result.mean(),
                                        result.median(),
                                        result.std(),
                                        result.min(),
                                        result.max()
                                    ]
                                })
                                
                                st.write("Statistics of new feature:")
                                st.dataframe(stats)
                                
                                # Visualisasi distribusi
                                fig = px.histogram(
                                    x=result,
                                    title=f"Distribution of {new_feature_name}",
                                    marginal="box"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Scatter plot dengan fitur asli
                                fig = px.scatter(
                                    x=df[feature1],
                                    y=df[feature2],
                                    color=result,
                                    title=f"Scatter Plot: {feature1} vs {feature2}, colored by {new_feature_name}",
                                    labels={'x': feature1, 'y': feature2, 'color': new_feature_name}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Korelasi dengan target (jika ada)
                                if 'Label' in df.columns and df['Label'].dtype != 'object':
                                    corr, _ = pearsonr(result.fillna(0), df['Label'])
                                    st.write(f"Correlation with target: {corr:.4f}")
                                    
                                    # Bandingkan dengan korelasi fitur asli
                                    corr1, _ = pearsonr(df[feature1], df['Label'])
                                    corr2, _ = pearsonr(df[feature2], df['Label'])
                                    
                                    corr_df = pd.DataFrame({
                                        'Feature': [feature1, feature2, new_feature_name],
                                        'Correlation with Target': [corr1, corr2, corr]
                                    })
                                    
                                    fig = px.bar(
                                        corr_df,
                                        x='Feature',
                                        y='Correlation with Target',
                                        title="Correlation Comparison",
                                        color='Correlation with Target',
                                        color_continuous_scale=px.colors.diverging.RdBu_r
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error in custom feature creation: {str(e)}")
                    else:
                        st.warning("Not enough numeric features for custom combinations")
                else:
                    st.warning("No numeric features found for feature engineering")
    
    return df, features

def train_anomaly_detection_model(df):
    """Train anomaly detection model"""
    st.subheader("Anomaly Detection Model Training")
    
    # Preprocess data
    processed_df, scaler = preprocess_for_anomaly_detection(df)
    
    # Enhanced visualization after training
    if st.checkbox("Show advanced visualizations"):
        # Correlation heatmap
        numeric_df = processed_df[numeric_cols]
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.title('Feature Correlation Heatmap')
        st.pyplot(fig)
    
    # Feature importance visualization
    if hasattr(results['model'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': results['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df.head(10), x='Feature', y='Importance',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig, use_container_width=True)

    # Select features for anomaly detection
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("No numeric features found for anomaly detection")
        return None
    
    # Select anomaly detection method
    method = st.selectbox(
        "Select anomaly detection method",
        options=["isolation_forest", "lof", "one_class_svm"]
    )
    
    # Set contamination parameter
    contamination = st.slider(
        "Contamination rate (expected proportion of anomalies)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01
    )
    
    # Train model
    if st.button("Train Anomaly Detection Model"):
        with st.spinner("Training anomaly detection model..."):
            # Prepare data
            X = processed_df[numeric_cols].values
            
            # Run anomaly detection
            results = run_anomaly_detection_pipeline(X, method, contamination)
            
            # Display results
            st.success("Anomaly detection model trained successfully!")
            
            # Show anomaly distribution
            anomaly_count = np.sum(results['anomaly_labels'])
            total_count = len(results['anomaly_labels'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", total_count)
                st.metric("Anomalies Detected", anomaly_count)
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_count/total_count:.2%}")
                st.metric("Normal Samples", total_count - anomaly_count)
            
            # Visualize results
            results_df = pd.DataFrame({
                'anomaly_score': results['anomaly_scores'],
                'is_anomaly': results['anomaly_labels']
            })
            
            # Distribution of anomaly scores
            fig = px.histogram(
                results_df, 
                x='anomaly_score',
                color='is_anomaly',
                title='Distribution of Anomaly Scores',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Save model
            model_data = {
                'model': results['model'],
                'scaler': scaler,
                'features': numeric_cols,
                'contamination': contamination,
                'method': method
            }
            
            model_path = os.path.join('d:\cybersecurity', 'anomaly_model.pkl')
            joblib.dump(model_data, model_path)
            st.success(f"Anomaly detection model saved to {model_path}")
            
            # Model persistence options
            st.subheader("Model Management")

            # Save model with metadata
            model_path = os.path.join('models', f'anomaly_model_{method}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl')
            saved_path = save_anomaly_model(
                results['model'], scaler, numeric_cols, method, contamination, model_path
            )
            st.success(f"Model saved to: {saved_path}")

            # List saved models
            if st.checkbox("View saved models"):
                models_dir = 'models'
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                    if model_files:
                        st.write("Available models:")
                        for model_file in model_files:
                            st.write(f"- {model_file}")
                    else:
                        st.info("No saved models found")
            return results
    return None

def train_model_page():
    st.title('Train Network Intrusion Detection Model')
    
    # File uploader for training data
    uploaded_file = st.file_uploader("Upload your training dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess data
        st.info("Loading and preprocessing data...")
        df = pd.read_csv(uploaded_file)
        
        # Display basic dataset info
        st.write("Dataset shape:", df.shape)  # Changed from processed_df to df
        st.write("Sample data:")
        st.dataframe(df.head())
        
        # Check if 'Label' column exists
        if 'Label' not in df.columns:  # Changed from processed_df to df
            st.error("Dataset must contain a 'Label' column for training.")
            return
        
        # EDA and feature extraction
        with st.expander("Exploratory Data Analysis", expanded=False):
            perform_eda(df)
        
        # Feature extraction and selection
        with st.expander("Feature Extraction and Selection", expanded=False):
            df, selected_features = extract_features(df, df.columns.tolist())
        
        # Model training parameters
        st.subheader("Model Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_channels = st.slider("Hidden Channels", 8, 256, 64, step=8)
            num_layers = st.slider("Number of GNN Layers", 1, 5, 2)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.5, step=0.1)
        
        with col2:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            num_epochs = st.slider("Number of Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 16, 256, 64, step=16)
        
        # Advanced parameters
        with st.expander("Advanced Parameters", expanded=False):
            weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.0001, format="%.5f")
            early_stopping = st.checkbox("Enable Early Stopping", value=True)
            patience = st.slider("Patience for Early Stopping", 5, 50, 10) if early_stopping else 0
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, step=0.05)
            use_class_weights = st.checkbox("Use Class Weights for Imbalanced Data", value=True)
        
        # Anomaly detection option
    use_anomaly_detection = st.checkbox("Use Anomaly Detection (Unsupervised)", value=False)
    
    if use_anomaly_detection:
        train_anomaly_detection_model(df)
        return
    
    # Start training
    if st.button("Train Model"):

            st.info("Starting model training process...")

            try:
                # Prepare data for GNN
                with st.spinner("Preparing data for GNN model..."):
                    # Make sure we have the necessary features
                    if not selected_features or len(selected_features) < 2:
                        st.warning("Not enough features selected. Using all numeric features.")
                        selected_features = df.select_dtypes(include=[np.number]).columns.tolist()  # Changed from processed_df to df
                        if 'Label' in selected_features:
                            selected_features.remove('Label')
                    
                    # Extract features and labels
                    X = df[selected_features].values
                    y = df['Label'].values
                    
                    # Convert labels to numeric if they're not already
                    if not pd.api.types.is_numeric_dtype(df['Label']):
                        st.info("Converting categorical labels to numeric values")
                        label_mapping = {label: i for i, label in enumerate(df['Label'].unique())}
                        y = df['Label'].map(label_mapping).values
                        
                        # Display the mapping
                        mapping_df = pd.DataFrame({
                            'Original Label': list(label_mapping.keys()),
                            'Numeric Value': list(label_mapping.values())
                        })
                        st.write("Label mapping:")
                        st.dataframe(mapping_df)
                    
                    # Normalize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Initialize the model
                    model = IDSGNNModel(
                        input_dim=X_scaled.shape[1],
                        hidden_channels=hidden_channels,
                        num_classes=len(np.unique(y)),
                        num_layers=num_layers,
                        dropout_rate=dropout_rate
                    )
                    
                    # Create a placeholder for metrics
                    metrics_container = st.empty()
                    
                    # Create a placeholder for the training plot
                    plot_container = st.empty()
                    
                    # Prepare data for training
                    train_data = model.prepare_data(X_scaled, y, validation_split=validation_split)
                    
                    # Calculate class weights if needed
                    class_weights = None
                    if use_class_weights:
                        class_weights = model.calculate_class_weights(y)
                        st.write("Class weights:", class_weights)
                    
                    # Train the model
                    training_results = model.train_model(
                        train_data,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        early_stopping=early_stopping,
                        patience=patience,
                        class_weights=class_weights,
                        progress_callback=lambda epoch, total, train_loss, val_loss, train_acc, val_acc: update_progress(
                            epoch, total, train_loss, val_loss, train_acc, val_acc,
                            progress_bar, metrics_container, plot_container
                        )
                    )
                    
                    # Display final results
                    st.success("Model training completed!")
                    
                    # Display training history
                    st.subheader("Training History")
                    history_df = pd.DataFrame(training_results['history'])
                    st.line_chart(history_df[['train_loss', 'val_loss']])
                    st.line_chart(history_df[['train_acc', 'val_acc']])
                    
                    # Display model evaluation
                    st.subheader("Model Evaluation")
                    evaluation = model.evaluate_model(train_data['test_data'], train_data['test_labels'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test Accuracy", f"{evaluation['accuracy']:.4f}")
                        st.metric("Precision", f"{evaluation['precision']:.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{evaluation['recall']:.4f}")
                        st.metric("F1 Score", f"{evaluation['f1']:.4f}")
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(evaluation['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    st.pyplot(fig)
                    
                    # Display classification report
                    st.subheader("Classification Report")
                    st.text(evaluation['classification_report'])
                    
                    # Save the model
                    model_path = os.path.join('d:\\cybersecurity', 'trained_model.pkl')
                    scaler_path = os.path.join('d:\\cybersecurity', 'feature_scaler.pkl')
                    
                    model.save_model(model_path)
                    joblib.dump(scaler, scaler_path)
                    
                    st.success(f"Model saved to {model_path}")
                    st.success(f"Feature scaler saved to {scaler_path}")
                    
                    # Save feature information
                    feature_info = {
                        'selected_features': selected_features,
                        'label_mapping': label_mapping if not pd.api.types.is_numeric_dtype(df['Label']) else None
                    }
                    
                    feature_info_path = os.path.join('d:\\cybersecurity', 'feature_info.pkl')
                    joblib.dump(feature_info, feature_info_path)
                    st.success(f"Feature information saved to {feature_info_path}")
                    
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.exception(e)

# Helper function to update progress during training
def update_progress(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, progress_bar, metrics_container, plot_container):
    # Update progress bar
    progress = (epoch + 1) / total_epochs
    progress_bar.progress(progress)
    
    # Update metrics
    metrics_container.markdown(f"""
    **Epoch {epoch+1}/{total_epochs}**
    - Training Loss: {train_loss:.4f}
    - Validation Loss: {val_loss:.4f}
    - Training Accuracy: {train_acc:.4f}
    - Validation Accuracy: {val_acc:.4f}
    """)
    
    # Update plot
    history_data = {
        'Epoch': list(range(1, epoch+2)),
        'Training Loss': [train_loss],
        'Validation Loss': [val_loss],
        'Training Accuracy': [train_acc],
        'Validation Accuracy': [val_acc]
    }
    
    # Create and update the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_data['Epoch'], y=history_data['Training Loss'], mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=history_data['Epoch'], y=history_data['Validation Loss'], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Training Progress', xaxis_title='Epoch', yaxis_title='Loss')
    
    plot_container.plotly_chart(fig)

def predict_anomalies(df, model_data):
    """Predict anomalies using trained model"""
    st.subheader("Anomaly Detection Results")
    
    # Preprocess data
    processed_df = df.copy()
    scaler = model_data['scaler']
    features = model_data['features']
    method = model_data['method']
    
    # Ensure all required features are present
    missing_features = [f for f in features if f not in processed_df.columns]
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        return None
    
    # Scale features
    X = processed_df[features].values
    X_scaled = scaler.transform(X)
    
    # Run anomaly detection
    results = run_anomaly_detection_pipeline(X_scaled, method, model_data['contamination'])
    
    # Add results to dataframe
    processed_df['anomaly_score'] = results['anomaly_scores']
    processed_df['is_anomaly'] = results['anomaly_labels']
    
    # Display results
    st.success("Anomaly detection completed!")
    
    # Summary statistics
    anomaly_count = np.sum(processed_df['is_anomaly'])
    total_count = len(processed_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", total_count)
        st.metric("Anomalies Detected", anomaly_count)
    with col2:
        st.metric("Anomaly Rate", f"{anomaly_count/total_count:.2%}")
    
    # Visualize results
    fig = px.histogram(
        processed_df,
        x='anomaly_score',
        color='is_anomaly',
        title='Distribution of Anomaly Scores',
        nbins=50
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top anomalies
    st.subheader("Top Anomalies")
    top_anomalies = processed_df.nlargest(10, 'anomaly_score')[features + ['anomaly_score']]
    st.dataframe(top_anomalies)
    
    # Download results
    csv = processed_df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )
    
    return processed_df

def predict_page():
    st.title('Predict Network Intrusions')
    
    # File upload section
    uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess the data
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Check if model exists
            if not os.path.exists('models/ids_model.pt'):
                st.error("No trained model found. Please train a model first.")
                return
            
            # Preprocess data
            data, features = preprocess_data(df)
            
            # Check if anomaly detection model exists
            anomaly_model_path = os.path.join('d:\cybersecurity', 'anomaly_model.pkl')
            
            if os.path.exists(anomaly_model_path):
                model_data = joblib.load(anomaly_model_path)
                
                if st.button("Detect Anomalies"):
                    results = predict_anomalies(df, model_data)
                    
                    if results is not None:
                        # Additional visualization
                        st.subheader("Feature Analysis for Anomalies")
                        
                        # Select features to visualize
                        numeric_cols = results.select_dtypes(include=[np.number]).columns.tolist()
                        if 'anomaly_score' in numeric_cols:
                            numeric_cols.remove('anomaly_score')
                        
                        if len(numeric_cols) > 0:
                            selected_feature = st.selectbox(
                                "Select feature to visualize",
                                options=numeric_cols
                            )
                            
                            # Box plot showing feature distribution by anomaly status
                            fig = px.box(
                                results,
                                x='is_anomaly',
                                y=selected_feature,
                                title=f'{selected_feature} Distribution by Anomaly Status'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No anomaly detection model found. Please train a model first.")
                
                # Fallback to original prediction method
                if os.path.exists('models/ids_model.pt'):
                    model = IDSGNNModel(input_dim=len(features))
                    model.load_state_dict(torch.load('models/ids_model.pt'))
                    model.eval()
                    
                    if st.button("Detect Intrusions"):
                        with torch.no_grad():
                            out = model(data.x, data.edge_index)
                            pred = out.argmax(dim=1)
                        
                        st.success("Analysis Complete!")
                        results_df = pd.DataFrame({
                            'Sample': range(len(pred)),
                            'Prediction': pred.numpy()
                        })
                        st.write("Prediction Results:")
                        st.dataframe(results_df)
                        
                        fig = px.histogram(results_df, x='Prediction', title='Distribution of Predictions')
                        st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please ensure your data is in the correct format and try again.")

def save_anomaly_model(model, scaler, features, method, contamination, filepath):
    """Save trained anomaly detection model"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'method': method,
        'contamination': contamination,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_data, filepath)
    return filepath

def load_anomaly_model(filepath):
    """Load saved anomaly detection model"""
    if os.path.exists(filepath):
        model_data = joblib.load(filepath)
        return model_data
    else:
        st.error(f"Model file not found: {filepath}")
        return None

# Main function to run the app
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Navigation
    if st.session_state['page'] == 'home':
        show_home_page()
    elif st.session_state['page'] == 'train':
        train_model_page()
    elif st.session_state['page'] == 'predict':
        predict_page()
    
# Run the app
if __name__ == '__main__':
    main()


