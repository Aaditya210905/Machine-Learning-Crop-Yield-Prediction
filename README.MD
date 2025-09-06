# 🌾 Crop Yield Prediction

A machine learning project that predicts agricultural crop yields based on environmental and agricultural factors using the CatBoost gradient boosting algorithm.

## 📋 Overview

This project leverages historical agricultural data to build a predictive model for crop yields across different Indian states. The model takes into account various factors including crop type, growing season, geographical location, rainfall patterns, and agricultural inputs to provide accurate yield predictions.

## 🎯 Features

- **Multi-factor Analysis**: Incorporates crop type, season, state, area, production, rainfall, fertilizer, and pesticide usage
- **Advanced ML Algorithm**: Utilizes CatBoost for robust gradient boosting with categorical feature support
- **Geographic Coverage**: Trained on data from multiple Indian states
- **Seasonal Predictions**: Supports different cropping seasons (Kharif, Rabi, Whole Year)
- **Easy-to-use Interface**: Simple Python scripts for training and prediction

## 📊 Dataset

The `crop_yield.csv` dataset contains comprehensive agricultural data with the following features:

### Input Features
- **Crop**: The specific type of crop cultivated
- **Crop_Year**: The year in which the crop was grown
- **Season**: The particular cropping season (Kharif, Rabi, or Whole Year)
- **State**: The Indian state where the crop was cultivated
- **Area**: The total land area under cultivation (in hectares)
- **Production**: The quantity of crop production (in metric tons)
- **Annual_Rainfall**: The total annual rainfall in the crop-growing region (in mm)
- **Fertiliser**: The total amount of fertiliser used for the crop (in kilograms)
- **Pesticide**: The total amount of pesticide used for the crop (in kilograms)

### Target Variable
- **Yield**: The calculated crop yield, expressed as production per unit area

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crop-yield-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```
catboost>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 💻 Usage

### Training the Model

Run the preprocessing and model training pipeline:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Split data into training and validation sets
- Train the CatBoost model with optimal hyperparameters
- Save the trained model for future predictions
- Display performance metrics

### Making Predictions

Predict yield for a new sample:

```bash
python src/predict_sample.py
```

### Interactive Analysis

Explore the data and model performance using the Jupyter notebook:

```bash
jupyter notebook notebooks/Crop_Yield_Model.ipynb
```

## 📈 Model Performance

The CatBoost model provides:
- High accuracy predictions with robust handling of categorical variables
- Feature importance analysis to understand key yield drivers
- Visualization of prediction vs actual yield patterns

## 🗂️ Project Structure

```
crop-yield-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── crop_yield.csv
├── src/
│   ├── train_model.py
│   ├── predict_sample.py
│   └── preprocessing.py
├── notebooks/
│   └── Crop_Yield_Model.ipynb
└──  models/
    └── catboost_model.pkl
    └── ohe_encoder.pkl
    └── pt_y.pkl
    └── pt.pkl
    └── target_encoder.pkl
```

## 🔍 Key Insights

- **Rainfall Impact**: Annual rainfall significantly influences crop yield across different regions
- **Fertilizer Efficiency**: Optimal fertilizer usage shows strong correlation with higher yields
- **Regional Variations**: Different states show varying yield patterns for the same crops
- **Seasonal Effects**: Kharif and Rabi seasons demonstrate distinct yield characteristics



## 👥 Author

- **Aaditya** 

## 🙏 Acknowledgments

- Indian agricultural department for providing comprehensive crop data
- CatBoost team for the excellent gradient boosting library
- Agricultural research communities for domain insights

⭐ **Star this repository if you find it helpful!**