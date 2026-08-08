# AgriShield - Core4 Project Progress

## Team Members
- Sahil
- Shraddha
- Sangamesh
- Sai

---

## Overall Project Status

- [x] Project planning and methodology
- [x] Literature survey
- [x] GitHub repository setup
- [x] Basic project folder structure
- [x] Development environment setup
- [x] Dataset finalization (Image)
- [x] Dataset finalization (Weather)
- [x] Image preprocessing
- [x] Weather data preprocessing
- [x] Swin Transformer training
- [x] Swin Transformer evaluation
- [x] LSTM training
- [x] LSTM evaluation
- [x] LightGBM training
- [x] LightGBM evaluation
- [ ] Hybrid prediction module
- [ ] Model integration
- [ ] Streamlit application
- [ ] Final system testing
- [ ] Documentation

---

## Sahil

**Current Responsibility:** AI Model Development (Image Detection & Weather Risk Prediction)

### Completed

#### Project Setup
- Created/shared GitHub repository
- Cloned repository locally
- Created initial project structure
- Completed initial Git commits and pushes
- Python virtual environment created
- PyTorch with CUDA configured
- RTX 4050 GPU successfully detected by PyTorch
- Installed all required ML libraries

#### Image Dataset
- Collected Corn Leaf Infection Dataset
- Verified dataset (4,225 images)
- Removed corrupted images
- Performed Train/Validation/Test split (70/15/15)

#### Image Preprocessing
- Created preprocessing pipeline
- Implemented image transforms
- Built PyTorch DataLoaders
- Verified tensor loading and batch processing

#### Swin Transformer
- Implemented Swin Transformer Tiny using TIMM
- Configured transfer learning
- Trained image classification model
- Saved best model (`models/swin_best.pth`)
- Developed evaluation script
- Achieved **99.21% test accuracy**

#### Weather Dataset
- Selected Fall Armyworm Weather Dataset
- Completed dataset exploration
- Performed feature selection
- Encoded categorical features
- Normalized numerical features
- Split dataset into training and testing sets

#### LSTM Model
- Implemented 2-layer LSTM model
- Created PyTorch DataLoaders
- Trained weather prediction model
- Evaluated model
- Saved trained model (`models/lstm_best.pth`)
- Achieved **64.00% test accuracy**

#### LightGBM Model
- Implemented LightGBM classifier
- Trained weather prediction model
- Evaluated model
- Generated feature importance analysis
- Saved trained model (`models/lightgbm_model.pkl`)
- Achieved **82.00% test accuracy**

---

### Current Task

- Hybrid prediction module (Swin Transformer + LightGBM integration)

### Next Task

- Develop Streamlit application
- Integrate image classification and weather prediction
- Perform complete system testing
- Final documentation and project report

---

## Shraddha

**Current Responsibility:** To be assigned

### Completed
- None recorded yet

### Current Task
- To be assigned

### Next Task
- To be assigned

---

## Sangamesh

**Current Responsibility:** Weather Forecasting & Risk Prediction

### Completed

- Finalized NASA POWER Mysuru weather dataset (2015–2025)
- Implemented complete weather preprocessing pipeline
- Built custom WeatherDataset for time-series sequence generation
- Developed multi-output LSTM forecasting model
- Forecasted Temperature (T2M), Humidity (RH2M), Wind Speed (WS2M), Surface Pressure (PS) and Rainfall (PRECTOTCORR)
- Implemented training and validation pipeline
- Saved best LSTM model (`models/lstm_best.pth`)
- Built evaluation pipeline with prediction CSVs and plots
- Designed weighted weather risk-score generation
- Generated balanced Low / Medium / High risk labels
- Trained multiclass LightGBM classifier
- Saved best LightGBM model (`models/lightgbm_best.pkl`)
- Generated classification report, confusion matrix and feature importance plot

### Results

- LSTM Temperature R²: 0.8848
- LSTM Humidity R²: 0.8977
- LSTM Wind Speed R²: 0.7859
- LSTM Pressure R²: 0.8263
- LSTM Rainfall R²: 0.1540
- LightGBM Accuracy: 98.15%

### Current Task

- Integrating weather forecasting with the crop disease detection module

### Next Task

- Hybrid system integration
- Streamlit dashboard
- End-to-end testing

## Sai

**Current Responsibility:** To be assigned

### Completed
- None recorded yet

### Current Task
- To be assigned

### Next Task
- To be assigned

---

## Current AgriShield Stage

Hybrid AI system development.

### Completed Models

- ✅ Swin Transformer (99.21% Accuracy)
- ✅ LSTM (64.00% Accuracy)
- ✅ LightGBM (82.00% Accuracy)

### Remaining Modules

- Hybrid Prediction Module
- Streamlit Web Application
- Final Integration
- System Testing
- Documentation