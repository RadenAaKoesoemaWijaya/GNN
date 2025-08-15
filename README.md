# GRAPHNET - API Log Anomaly Detection System

A sophisticated machine learning application for detecting anomalies in API logs using Autoencoder and Graph Neural Networks (GNN) techniques.

## Overview

GRAPHNET is a comprehensive anomaly detection system designed specifically for API log analysis. It combines the power of Autoencoders for feature learning and Graph Neural Networks for relationship modeling to identify unusual patterns in API usage.

## System Workflow

The application follows a systematic 6-step workflow:

### 1. Data Collection
Upload your API log data in CSV format. The system supports logs containing:
- **IP Address**: Source address of requests
- **Timestamp**: Time of API calls
- **User Agent**: Client/application information
- **Endpoint/API Call**: Specific API endpoints accessed
- **Parameters**: Request parameters
- **Response Status Code**: HTTP status codes (200, 404, 500, etc.)
- **Response Time**: API response duration

### 2. Data Preprocessing
Automatic feature extraction and data preparation:
- **Timestamp Feature Extraction**: Hour, day, day-of-week from timestamps
- **IP Address Analysis**: Extract octets and network information
- **User Agent Parsing**: Browser, OS, and device detection
- **Endpoint Encoding**: One-hot encoding for categorical endpoints
- **Status Code Categorization**: Success, redirect, client error, server error
- **Time-based Aggregation**: Request frequency per IP/time intervals
- **Data Normalization**: StandardScaler for numerical features
- **Missing Value Handling**: Fill with zeros for robust processing

### 3. Graph Formation
Transform API logs into graph structures for GNN analysis:
- **Node Creation**: Each log entry becomes a node
- **Edge Formation**: Connections based on IP addresses and endpoints
- **Sequential Graphs**: Fallback to sequential connections when specific columns aren't found
- **Feature Assignment**: Node features from preprocessed data

### 4. Model Training
Dual-model training approach:
- **Autoencoder Training**: 
  - Unsupervised learning on normal data
  - Feature reconstruction for anomaly detection
  - Configurable architecture (input_size, hidden_size, latent_size)
  - Early stopping and validation monitoring
- **GNN Training**:
  - Graph-based learning for relationship patterns
  - GCNConv layers for message passing
  - Global mean pooling for graph-level representations
  - Combined with autoencoder features for enhanced detection

### 5. Anomaly Detection
Real-time anomaly identification:
- **Threshold Calculation**: Dynamic threshold based on reconstruction errors
- **Multi-model Scoring**: Combined autoencoder and GNN predictions
- **Anomaly Scoring**: Probability scores for each log entry
- **False Positive Reduction**: Advanced filtering techniques

### 6. Result Evaluation
Comprehensive analysis and reporting:
- **Visualization**: Interactive plots for anomalies and patterns
- **Metrics**: Precision, recall, F1-score, ROC-AUC
- **Export Options**: Downloadable anomaly reports
- **Real-time Monitoring**: Continuous anomaly detection on new data

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
# Clone the repository
git clone [repository-url]
cd GNN

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Navigate Through the Workflow
- **Home Page**: Overview and navigation
- **Data Collection**: Upload CSV files and map columns
- **Preprocessing**: Automatic feature extraction
- **Training**: Configure and train models
- **Detection**: Run anomaly detection on new data

### 3. Input Data Format
Your CSV should contain columns with these keywords (case-insensitive):
- **IP**: 'ip', 'address', 'src', 'source'
- **Timestamp**: 'time', 'date', 'timestamp'
- **User Agent**: 'user', 'agent', 'browser', 'ua'
- **Endpoint**: 'endpoint', 'api', 'url', 'path'
- **Status**: 'status', 'code', 'response_code'

## Technical Architecture

### Models
- **APILogAutoencoder**: Custom autoencoder for log data
- **IDSGNNModel**: Graph neural network for API relationships

### Dependencies
- **Streamlit**: Web interface
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural networks
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Plotly**: Interactive visualizations

### Model Storage
- `models/autoencoder.pt`: Trained autoencoder weights
- `models/gnn_model.pt`: Trained GNN weights
- `models/anomaly_threshold.txt`: Detection threshold
- `models/*.json`: Model parameters and configurations

## Configuration

### Model Parameters
- **Autoencoder**: Configurable hidden layers and latent space
- **GNN**: Adjustable graph convolution layers
- **Training**: Customizable epochs, learning rate, batch size

### Web Interface
- Responsive design with custom CSS
- Real-time progress tracking
- Interactive visualizations
- Export capabilities

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all packages in requirements.txt are installed
2. **Data Format**: Verify CSV format matches expected structure
3. **Memory Issues**: Reduce batch size for large datasets
4. **Training Time**: Adjust epochs based on dataset size

### Error Handling
- Comprehensive error messages in the UI
- Graceful fallback for missing columns
- Automatic data validation and cleaning

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest

# Code formatting
black .
isort .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review error messages in the application
- Ensure data format compatibility
- Verify all dependencies are properly installed

## Performance Notes

- **Scalability**: Handles datasets from thousands to millions of records
- **Memory Usage**: Optimized for standard hardware configurations
- **Training Time**: Varies based on dataset size and model complexity
- **Real-time Processing**: Supports continuous monitoring scenarios



