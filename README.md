# AgriShield-FAW 🌾

### Hybrid System for Fall Armyworm Detection and Outbreak Prediction in Maize Crops

AgriShield-FAW is a machine learning-based system designed to assist in the **early detection of Fall Armyworm (FAW)** and **prediction of potential outbreaks in maize crops**.

The project combines **computer vision, weather forecasting, and machine learning** to provide a hybrid approach for crop protection. The system is being developed with separate pipelines for image-based FAW detection, weather prediction, and outbreak prediction.

---

## 🎯 Project Objective

Fall Armyworm is a major threat to maize production and can cause significant crop damage if not detected early.

AgriShield aims to provide a data-driven solution that can:

* Detect Fall Armyworm from crop images.
* Analyze environmental and weather conditions.
* Predict weather patterns that may influence pest outbreaks.
* Support early identification of potential FAW outbreaks.
* Provide prediction results through an integrated application interface.

---

## 🏗️ System Overview

AgriShield combines multiple machine learning components:

```text
                ┌─────────────────────┐
                │     Crop Images     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Swin Transformer   │
                │   FAW Detection     │
                └──────────┬──────────┘
                           │
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│              AgriShield Prediction              │
│                                                 │
│  Weather Data ──► LSTM ──► Weather Prediction  │
│                                                 │
│  Environmental Data ──► LightGBM ──►           │
│                         Outbreak Prediction     │
└─────────────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prediction Results  │
                └─────────────────────┘
```

---

## 🤖 Machine Learning Components

### 1. Swin Transformer

A **Swin Transformer** based computer vision pipeline is used for image-based FAW detection and classification.

The pipeline is intended to identify FAW-related patterns from maize crop images and provides the foundation for image-based pest detection.

### 2. LSTM

An **LSTM (Long Short-Term Memory)** based model is used for weather prediction.

LSTM is suitable for sequential data and is used to learn patterns in historical weather information for predicting future weather conditions.

### 3. LightGBM

**LightGBM** is used for structured-data prediction.

The project uses a LightGBM-based pipeline for analyzing environmental/weather-related information and supporting **FAW outbreak prediction**.

---

## 🧩 Project Architecture

The repository is organized into separate components for data processing, model development, training, evaluation, and application integration.

```text
AgriShield-FAW/
│
├── app/                    # Application/backend components
│
├── frontend/               # Frontend interface
│
├── data/                   # Project datasets
│
├── database/               # FAW database and related data
│
├── models/                 # Trained/model-related files
│
├── notebooks/              # Development and experimentation notebooks
│
├── preprocessing/          # Data preprocessing pipelines
│
├── training/               # Model training pipelines
│
├── evaluation/             # Model evaluation
│
├── utils/                  # Utility functions
│
├── requirements.txt        # Python dependencies
│
├── test_imports.py         # Import/environment tests
│
├── test_sklearn.py         # ML environment tests
│
└── README.md
```

---

## 🔬 Data Pipeline

The general machine learning workflow follows:

```text
Raw Data
   │
   ▼
Data Preprocessing
   │
   ▼
Feature / Image Preparation
   │
   ├───────────────┐
   │               │
   ▼               ▼
Image Data      Weather /
   │            Environmental Data
   │               │
   ▼               ▼
Swin            LSTM / LightGBM
Transformer
   │               │
   ▼               ▼
FAW Detection   Prediction
   │               │
   └───────┬───────┘
           ▼
    AgriShield Results
```

---

## 🛠️ Technologies Used

### Machine Learning

* Python
* PyTorch
* Swin Transformer
* LightGBM
* LSTM
* Scikit-learn

### Data Processing

* NumPy
* Pandas
* DataLoaders
* Image preprocessing

### Application

* Python backend/application components
* HTML
* CSS
* JavaScript

### Development Tools

* Git
* GitHub
* Jupyter Notebook
* VS Code

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sahilllm-cloud/AgriShield-FAW.git
cd AgriShield-FAW
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

The individual training, evaluation, and application components can be executed according to the scripts and notebooks provided in their respective directories.

---

## 📊 Current Development

The project is currently under active development.

### Completed / In Progress

* [x] Initial project structure
* [x] FAW dataset/database integration
* [x] Image preprocessing pipeline
* [x] Swin Transformer training pipeline
* [x] LightGBM training pipeline
* [x] Weather prediction pipeline
* [x] LSTM-based weather modeling
* [x] Model evaluation structure
* [x] Frontend/backend integration work
* [ ] Complete end-to-end prediction workflow
* [ ] Final model evaluation and benchmarking
* [ ] Production deployment

---

## 🔮 Future Improvements

Potential future improvements include:

* Real-time crop image analysis
* Improved FAW detection accuracy
* Integration of live weather data
* More robust outbreak forecasting
* Mobile-friendly interface
* Farmer-oriented alerts and recommendations
* Deployment of trained models as scalable APIs
* Continuous model improvement using new field data

---

## 📌 Project Status

**Active Development 🚧**

AgriShield-FAW is being developed as a hybrid machine learning system combining **computer vision, time-series forecasting, and structured-data prediction** to support early Fall Armyworm detection and outbreak prediction in maize crops.
