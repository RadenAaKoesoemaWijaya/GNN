# Network Intrusion Detection System

A Graph Neural Network-based application for detecting network intrusions in network traffic data.

## Overview

This application uses Graph Neural Networks (GNNs) to analyze network traffic data and identify potential intrusions or attacks. It provides a user-friendly interface built with Streamlit that allows users to:

- Upload network traffic data in CSV format
- Visualize network traffic patterns
- Train new detection models or use existing ones
- Analyze detection results with detailed visualizations

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone or download this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```
   streamlit run app.py
   ```

## Usage

### Getting Started

1. After launching the application, you'll be presented with the home page showing two options:
   - **Train New Model**: Upload data, perform exploratory data analysis (EDA), and train a new GNN model
   - **Use Existing Model**: Upload data and use a pre-trained model to detect intrusions

### Training a New Model

1. Click on the **Train New Model** button on the home page.
2. Upload your network traffic data in CSV format.
3. The application will automatically preprocess your data and perform the following steps:
   - Data cleaning (removing timestamp columns, handling IP addresses)
   - Label encoding for classification
   - Handling missing values and outliers
   - Feature selection and normalization

4. Explore your data through the comprehensive Exploratory Data Analysis (EDA) tools:
   - **Data Overview**: View basic statistics, class distribution, and missing values analysis
   - **Feature Analysis**: Examine feature distributions and outlier detection
   - **Correlation Analysis**: Identify relationships between features using correlation matrices
   - **Distribution Analysis**: Compare feature distributions across different classes
   - **Dimensionality Reduction**: Visualize data using PCA or t-SNE

5. Configure and train your GNN model:
   - Set hyperparameters (learning rate, hidden layers, etc.)
   - Choose training parameters (batch size, epochs)
   - Monitor training progress with real-time metrics

6. Evaluate model performance:
   - View accuracy, precision, recall, and F1-score metrics
   - Examine the confusion matrix
   - Analyze feature importance

7. Save your trained model for future use.

### Using an Existing Model

1. Click on the **Use Existing Model** button on the home page.
2. Upload your network traffic data for analysis.
3. Select a previously trained model.
4. The application will process your data and apply the selected model to detect intrusions.
5. Explore the detection results:
   - View predicted intrusion types
   - Analyze detection confidence scores
   - Visualize network traffic patterns with highlighted intrusions
   - Examine detailed information about detected threats

### Supported Attack Types

The system can detect various types of network attacks, including:
- DDoS/DoS attacks
- Port scanning
- Brute force attacks
- Web attacks (SQL injection, XSS)
- Botnet activities
- And more...

### Data Requirements

For optimal performance, your network traffic data should include:
- Network connection information (source/destination IPs, ports)
- Traffic statistics (packet sizes, counts, intervals)
- Protocol information
- Connection states

The application will automatically preprocess your data, but having clean, well-structured data will improve detection accuracy.



