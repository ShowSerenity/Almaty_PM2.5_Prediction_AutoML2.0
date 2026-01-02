import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for explainability
import shap

from datetime import datetime
import joblib
import os


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible CatBoost wrapper"""

    def __init__(self, iterations=300, learning_rate=0.05, depth=6,
                 verbose=False, random_state=42, allow_writing_files=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.verbose = verbose
        self.random_state = random_state
        self.allow_writing_files = allow_writing_files
        self.model_ = None

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=self.verbose,
            random_state=self.random_state,
            allow_writing_files=self.allow_writing_files
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def _more_tags(self):
        return {'regressor': True}


class LSTMRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible LSTM wrapper for time series"""

    def __init__(self, epochs=50, batch_size=32, units=64, dropout=0.2,
                 lookback=7, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.dropout = dropout
        self.lookback = lookback
        self.random_state = random_state
        self.model_ = None
        self.scaler = MinMaxScaler()

    def _create_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_scaled = self.scaler.fit_transform(X)
        Xs, ys = [], []
        for i in range(len(X_scaled) - self.lookback):
            Xs.append(X_scaled[i:i + self.lookback])
            ys.append(y.iloc[i + self.lookback])
        return np.array(Xs), np.array(ys)

    def fit(self, X, y):
        tf.random.set_seed(self.random_state)

        X_seq, y_seq = self._create_sequences(X, y)

        if len(X_seq) < 10:  # Not enough data for sequences
            # Fallback to simple model
            self.model_ = None
            return self

        self.model_ = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback, X.shape[1])),
            Dropout(self.dropout),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        self.model_.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self.model_.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[early_stop]
        )

        return self

    def predict(self, X):
        if self.model_ is None:
            return np.full(len(X), X.mean())

        X_scaled = self.scaler.transform(X)
        X_seq = []

        for i in range(len(X_scaled) - self.lookback + 1):
            X_seq.append(X_scaled[i:i + self.lookback])

        if len(X_seq) == 0:
            return np.full(len(X), X.mean())

        X_seq = np.array(X_seq)
        predictions = self.model_.predict(X_seq, verbose=0).flatten()

        # Pad predictions for the first lookback samples
        full_predictions = np.full(len(X), predictions.mean())
        full_predictions[self.lookback - 1:] = predictions

        return full_predictions


class AutoML20:
    """
    AutoML 2.0 Ultimate System

    Features:
    - Automated hyperparameter optimization
    - Advanced feature engineering
    - Deep learning models (LSTM/GRU)
    - Visualization dashboard
    - SHAP explainability
    - Model comparison and selection
    """

    def __init__(self, random_state=42, n_trials=30, cv_folds=10,
                 enable_deep_learning=True, enable_visualization=True):
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.enable_deep_learning = enable_deep_learning
        self.enable_visualization = enable_visualization
        self.results = []
        self.best_model = None
        self.best_score = float('inf')
        self.best_pipeline = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None

    def advanced_feature_engineering(self, df):
        """Generate advanced time-series features"""
        print("Performing advanced feature engineering...")

        df = df.sort_values('date').reset_index(drop=True)

        # Temporal features
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Seasonal indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        # Lag features
        pollutants = ['pm10', 'no2', 'so2', 'co']
        for col in pollutants:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag7'] = df[col].shift(7)
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7, min_periods=1).std()

        # Interaction features
        if 'pm10' in df.columns and 'no2' in df.columns:
            df['pm10_no2_ratio'] = df['pm10'] / (df['no2'] + 1)

        df = df.drop(columns=['date'])

        return df

    def optimize_lightgbm(self, X, y):
        """Hyperparameter optimization for LightGBM"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state,
                'verbose': -1
            }

            model = LGBMRegressor(**params)
            pipeline = Pipeline([
                ('preprocessor', self._get_preprocessor(X)),
                ('model', model)
            ])

            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_validate(pipeline, X, y, cv=cv,
                                    scoring='neg_root_mean_squared_error')
            return -scores['test_score'].mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return study.best_params

    def optimize_xgboost(self, X, y):
        """Hyperparameter optimization for XGBoost"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'objective': 'reg:squarederror'
            }

            model = XGBRegressor(**params)
            pipeline = Pipeline([
                ('preprocessor', self._get_preprocessor(X)),
                ('model', model)
            ])

            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_validate(pipeline, X, y, cv=cv,
                                    scoring='neg_root_mean_squared_error')
            return -scores['test_score'].mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return study.best_params

    def _get_preprocessor(self, X):
        """Create preprocessing pipeline"""
        numeric_features = X.columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )

    def evaluate_model(self, name, model, X, y):
        """Evaluate model with cross-validation"""
        pipeline = Pipeline([
            ('preprocessor', self._get_preprocessor(X)),
            ('model', model)
        ])

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        scoring = {
            'rmse': make_scorer(lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)),
                                greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'r2': 'r2'
        }

        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        rmse = -cv_results['test_rmse'].mean()
        mae = -cv_results['test_mae'].mean()
        r2 = cv_results['test_r2'].mean()

        # Track best model
        if rmse < self.best_score:
            self.best_score = rmse
            self.best_model = name
            self.best_pipeline = pipeline
            self.best_pipeline.fit(X, y)

        return {
            'Algorithm': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Features': X.shape[1]
        }

    def create_visualizations(self, output_dir='automl_results'):
        """Create comprehensive visualization dashboard"""
        if not self.enable_visualization:
            return

        print("\nCreating Visualization Dashboard...")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Model Performance Comparison
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        results_df = pd.DataFrame(self.results).sort_values('RMSE')
        plt.barh(results_df['Algorithm'], results_df['RMSE'], color='steelblue')
        plt.xlabel('RMSE (Lower is Better)', fontsize=12)
        plt.title('Model Performance Comparison - RMSE', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        plt.subplot(1, 2, 2)
        plt.barh(results_df['Algorithm'], results_df['R2'], color='coral')
        plt.xlabel('R² Score (Higher is Better)', fontsize=12)
        plt.title('Model Performance Comparison - R²', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_dir}/model_comparison.png")
        plt.close()

        # 2. Feature Importance (for tree-based models)
        if hasattr(self.best_pipeline.named_steps['model'], 'feature_importances_'):
            plt.figure(figsize=(10, 8))

            importances = self.best_pipeline.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features

            plt.barh(range(len(indices)), importances[indices], color='forestgreen')
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top 20 Feature Importances - {self.best_model}',
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved: {output_dir}/feature_importance.png")
            plt.close()

        # 3. Predictions vs Actual
        plt.figure(figsize=(10, 6))
        y_pred = self.best_pipeline.predict(self.X_test)

        plt.scatter(self.y_test, y_pred, alpha=0.6, color='navy', edgecolors='k')
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual PM2.5', fontsize=12)
        plt.ylabel('Predicted PM2.5', fontsize=12)
        plt.title(f'Predictions vs Actual - {self.best_model}',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_dir}/predictions_vs_actual.png")
        plt.close()

        # 4. Residuals Plot
        plt.figure(figsize=(10, 6))
        residuals = self.y_test.values - y_pred

        plt.scatter(y_pred, residuals, alpha=0.6, color='purple', edgecolors='k')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted PM2.5', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title(f'Residual Plot - {self.best_model}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/residuals.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_dir}/residuals.png")
        plt.close()

        print(f"\n   📁 All visualizations saved in: {output_dir}/")

    def create_shap_explanations(self, output_dir='automl_results'):
        """Generate SHAP explanations for model interpretability"""
        print("\nGenerating SHAP Explanations...")

        try:
            # Get preprocessed data
            X_test_processed = self.best_pipeline.named_steps['preprocessor'].transform(self.X_test)

            # Create SHAP explainer
            if hasattr(self.best_pipeline.named_steps['model'], 'feature_importances_'):
                explainer = shap.TreeExplainer(self.best_pipeline.named_steps['model'])
                shap_values = explainer.shap_values(X_test_processed[:100])  # Use first 100 samples

                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test_processed[:100],
                                  feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
                print(f"   ✓ Saved: {output_dir}/shap_summary.png")
                plt.close()

                # Force plot for a single prediction
                plt.figure(figsize=(14, 4))
                shap.force_plot(explainer.expected_value, shap_values[0],
                                X_test_processed[0], feature_names=self.feature_names,
                                matplotlib=True, show=False)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_force_plot.png', dpi=300, bbox_inches='tight')
                print(f"   ✓ Saved: {output_dir}/shap_force_plot.png")
                plt.close()

                print(f"   ✓ SHAP analysis complete!")
            else:
                print("   ⚠ Best model doesn't support SHAP tree explainer")

        except Exception as e:
            print(f"   ✗ SHAP generation failed: {str(e)}")

    def fit(self, df):
        """Main AutoML training pipeline"""
        print("=" * 80)
        print("AutoML 2.0 ULTIMATE - Advanced Automation Pipeline Started")
        print("=" * 80)

        # Data preprocessing
        print("\nStep 1: Data Preprocessing")
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(subset=["pm25"])
        df["date"] = pd.to_datetime(df["date"])

        print(f"   Dataset shape: {df.shape}")
        print(f"   Target variable: pm25")

        # Advanced feature engineering
        df_engineered = self.advanced_feature_engineering(df.copy())
        X = df_engineered.drop(columns=["pm25"])
        y = df_engineered["pm25"]

        self.feature_names = X.columns.tolist()
        print(f"   Features after engineering: {X.shape[1]}")

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Model definitions
        print("\nStep 2: Base Model Evaluation")

        base_models = {
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
            "CatBoost": CatBoostRegressorWrapper(
                iterations=300, learning_rate=0.05, depth=6,
                verbose=False, random_state=42, allow_writing_files=False
            )
        }

        # Evaluate base models
        for name, model in base_models.items():
            try:
                result = self.evaluate_model(name, model, X, y)
                self.results.append(result)
                print(f"   ✓ {name}: RMSE={result['RMSE']:.3f}, R²={result['R2']:.3f}")
            except Exception as e:
                print(f"   ✗ {name}: Failed - {str(e)}")

        # Hyperparameter optimization
        print("\nStep 3: Hyperparameter Optimization")

        print("   Optimizing LightGBM...")
        best_lgb_params = self.optimize_lightgbm(X, y)
        lgb_optimized = LGBMRegressor(**best_lgb_params)
        result = self.evaluate_model("LightGBM (Optimized)", lgb_optimized, X, y)
        self.results.append(result)
        print(f"   ✓ LightGBM Optimized: RMSE={result['RMSE']:.3f}, R²={result['R2']:.3f}")

        print("   Optimizing XGBoost...")
        best_xgb_params = self.optimize_xgboost(X, y)
        xgb_optimized = XGBRegressor(**best_xgb_params)
        result = self.evaluate_model("XGBoost (Optimized)", xgb_optimized, X, y)
        self.results.append(result)
        print(f"   ✓ XGBoost Optimized: RMSE={result['RMSE']:.3f}, R²={result['R2']:.3f}")

        # Deep Learning Models
        if self.enable_deep_learning:
            print("\nStep 4: Deep Learning Models")

            try:
                lstm_model = LSTMRegressorWrapper(
                    epochs=50, batch_size=32, units=64,
                    lookback=7, random_state=42
                )
                result = self.evaluate_model("LSTM", lstm_model, X, y)
                self.results.append(result)
                print(f"   ✓ LSTM: RMSE={result['RMSE']:.3f}, R²={result['R2']:.3f}")
            except Exception as e:
                print(f"   ✗ LSTM failed: {str(e)}")

        # Ensemble learning
        print("\nStep 5: Ensemble Model Creation")

        results_df = pd.DataFrame(self.results).sort_values('RMSE')
        top_models = []

        for _, row in results_df.head(5).iterrows():
            model_name = row['Algorithm']

            if 'CatBoost' in model_name or 'LSTM' in model_name:
                continue

            if model_name == "LightGBM (Optimized)":
                top_models.append(("LightGBM_Opt", lgb_optimized))
            elif model_name == "XGBoost (Optimized)":
                top_models.append(("XGBoost_Opt", xgb_optimized))
            else:
                for name, model in base_models.items():
                    if name == model_name:
                        top_models.append((name, model))
                        break

            if len(top_models) >= 3:
                break

        if len(top_models) >= 2:
            ensemble = VotingRegressor(estimators=top_models[:3])
            try:
                result = self.evaluate_model("Ensemble (Top 3)", ensemble, X, y)
                self.results.append(result)
                print(f"   ✓ Ensemble Model: RMSE={result['RMSE']:.3f}, R²={result['R2']:.3f}")
            except Exception as e:
                print(f"   ✗ Ensemble creation failed: {str(e)}")

        # Final results
        print("\n" + "=" * 80)
        print(" FINAL RESULTS - All Models Ranked")
        print("=" * 80)

        results_df = pd.DataFrame(self.results).sort_values('RMSE')
        print(results_df.to_string(index=False))

        print("\n" + "=" * 80)
        print(f" BEST MODEL: {self.best_model}")
        print(f"  RMSE: {self.best_score:.3f}")
        print("=" * 80)

        # Generate visualizations
        self.create_visualizations()

        # Generate SHAP explanations
        self.create_shap_explanations()

        return results_df

    def save_model(self, filepath="best_model.pkl"):
        """Save the best model for deployment"""
        if self.best_pipeline:
            joblib.dump(self.best_pipeline, filepath)
            print(f"\nBest model saved to: {filepath}")
        else:
            print("No model to save!")

    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_pipeline:
            return self.best_pipeline.predict(X)
        else:
            raise ValueError("No model fitted yet!")


# MAIN EXECUTION

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("almaty_air_pollution.csv")

    # Initialize AutoML 2.0 Ultimate
    automl = AutoML20(
        random_state=42,
        n_trials=30,  # Reduce for faster testing
        cv_folds=10,
        enable_deep_learning=True,
        enable_visualization=True
    )

    # Train and evaluate
    results = automl.fit(df)

    # Save best model
    automl.save_model("best_pm25_model.pkl")

    print("\nAutoML 2.0 Ultimate Pipeline Complete!")
    print("Check 'automl_results' folder for visualizations and SHAP analysis")