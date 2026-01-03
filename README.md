# AutoML 2.0 for PM2.5 Prediction in Almaty

## Project Overview
This project implements an AutoML 2.0 system for automated training, optimization, and evaluation of machine learning models to predict PM2.5 air pollution levels in Almaty, Kazakhstan. The system compares classical regression models, ensemble methods, boosting algorithms, and deep learning approaches under a unified framework with consistent preprocessing and evaluation.

The project was developed as part of a Master’s-level assignment and follows strict methodological requirements, including the use of 10-fold cross-validation and comprehensive performance comparison.

## Dataset
The dataset contains daily air pollution measurements for Almaty and includes the following variables:

- Date  
- PM2.5 (target variable)  
- PM10  
- NO₂  
- SO₂  
- CO  

Rows with missing PM2.5 values were removed. Missing values in predictor variables were handled using median imputation.

**Source:**  
Kaggle – Almaty Air Pollution Dataset  
https://www.kaggle.com/datasets/dauletb01/almaty-air-pollution-aqi

## Feature Engineering
The original dataset was expanded from 5 variables to 30 engineered features, including:

- Temporal features (month, day, weekday, seasonal indicators)  
- Lag features (1-day and 7-day lags for pollutants)  
- Rolling statistics (7-day mean and standard deviation)  
- Interaction feature (PM10 / NO₂ ratio)  

This approach captures temporal dependencies, seasonal effects, and pollutant interactions relevant to PM2.5 prediction.

## Models Implemented
The AutoML system evaluates the following regression models:

- Ridge Regression  
- Lasso Regression  
- Elastic Net  
- K-Nearest Neighbors Regression  
- Extra Trees Regression  
- AdaBoost Regression  
- Gradient Boosting Regression  
- HistGradientBoosting Regression  
- XGBoost (Optuna-optimized)  
- LightGBM (Optuna-optimized)  
- CatBoost  
- LSTM (time-series deep learning model)  
- Ensemble model (Voting Regressor of top-performing models)

All models are trained and evaluated using the same preprocessing pipeline.

## Evaluation Methodology
Model performance is assessed using 10-fold cross-validation. For each fold, models are trained on 90% of the data and tested on the remaining 10%. Final results are reported as the average across all folds.

Evaluation metrics:
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² score  


## Install dependencies:
```pip install -r requirements.txt```


The best-performing model will be saved automatically, and all visualizations will be stored in the `automl_results` folder.

## Results
The system outputs a ranked comparison of all models based on RMSE and R². Ensemble and boosting-based models generally outperform linear models, demonstrating the effectiveness of automated feature engineering and model optimization for air quality prediction.

## Notes
- This project addresses a regression task only.
- PM2.5 is the single target variable.
- All models are evaluated under identical conditions.
- The framework can be extended to other datasets or prediction targets.
