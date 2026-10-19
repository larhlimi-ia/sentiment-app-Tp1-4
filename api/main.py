"""
SCAF Main Framework - Complete Implementation
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from models.classical_models import ARIMA_Model, GARCH_Model, VAR_Model, EGARCH_Model
from models.ml_models import XGBoost_Model, RandomForest_Model, LightGBM_Model
from models.dl_models import LSTM_Model, PatchTST_Model
from orchestrator.q_learning import QLearningOrchestrator
from orchestrator.regime_detector import RegimeDetector
from aggregation.weighted_aggregation import WeightedAggregator
from aggregation.conformal_prediction import ConformalPredictor
from features.feature_engineering import FeatureEngineer
from evaluation.metrics import MetricsCalculator

class SCAF_Framework:
    """Complete SCAF Framework Implementation"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        np.random.seed(self.config.RANDOM_SEED)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(n_lags=self.config.N_LAGS)
        self.regime_detector = RegimeDetector(device=self.config.DEVICE)
        self.orchestrator = QLearningOrchestrator(
            n_models=self.config.N_EXPERTS,
            n_regimes=self.config.N_REGIMES,
            alpha=self.config.Q_LEARNING_ALPHA,
            gamma=self.config.Q_LEARNING_GAMMA,
            epsilon=self.config.Q_LEARNING_EPSILON
        )
        self.aggregator = WeightedAggregator(
            weight_uncertainty=self.config.WEIGHT_UNCERTAINTY,
            weight_performance=self.config.WEIGHT_PERFORMANCE,
            weight_q_values=self.config.WEIGHT_Q_VALUES
        )
        self.conformal_predictor = ConformalPredictor(alpha=self.config.CONFORMAL_ALPHA)
        
        # Initialize expert models
        self.experts = {
            'ARIMA': ARIMA_Model(),
            'GARCH': GARCH_Model(),
            'VAR': VAR_Model(),
            'EGARCH': EGARCH_Model(),
            'XGBoost': XGBoost_Model(),
            'RandomForest': RandomForest_Model(),
            'LightGBM': LightGBM_Model(),
            'LSTM': LSTM_Model(),
            'PatchTST': PatchTST_Model(),
        }
        
        # Tracking
        self.model_performance = {}
        self.is_fitted = False
    
    def prepare_data(self, prices: np.ndarray):
        """Prepare data for training"""
        # Create features
        features = self.feature_engineer.create_features(prices)
        
        # Create target (next day price)
        y = prices[1:].flatten()
        X = features[:-1]
        y = y[:len(X)]
        
        return X, y
    
    def fit(self, prices: np.ndarray, verbose=True):
        """Fit all components of SCAF"""
        if verbose:
            print("=" * 60)
            print("SCAF Framework - Training")
            print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_data(prices)
        
        # Train regime detector
        if verbose:
            print("\n📊 Training Regime Detector...")
        self.regime_detector.fit(X, epochs=100)
        
        # Train expert models
        if verbose:
            print("\n🔄 Training Expert Models...")
        
        for name, model in tqdm(self.experts.items(), desc="Training Models"):
            try:
                model.fit(X, y)
                if verbose:
                    status = "✓" if model.is_fitted else "✗"
                    print(f"  {status} {name}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {name}: {e}")
        
        self.is_fitted = True
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete")
            print("=" * 60)
    
    def predict(self, prices: np.ndarray, verbose=True) -> dict:
        """Make predictions using SCAF"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data
        X, y = self.prepare_data(prices)
        
        # Split into train/calibration/test for conformal prediction
        n_calib = int(len(X) * self.config.CONFORMAL_CALIBRATION_RATIO)
        X_calib, X_test = X[:n_calib], X[n_calib:]
        y_calib, y_test = y[:n_calib], y[n_calib:]
        
        predictions = []
        actuals = []
        uncertainties = []
        regimes_log = []
        weights_log = []
        
        if verbose:
            print("\n🔮 Making Predictions...")
        
        for t in tqdm(range(len(X_test)), desc="Predicting", disable=not verbose):
            # Get current features
            current_X = X_test[t:t+1]
            current_y = y_test[t]
            
            # Detect regime
            regime = self.regime_detector.detect_regime(current_X)
            
            # Select models
            available_models = [name for name, model in self.experts.items() if model.is_fitted]
            selected_models = self.orchestrator.select_models(regime, available_models, n_select=3)
            
            # Get predictions from selected models
            preds = []
            uncs = []
            perfs = []
            q_vals = []
            
            for model_name in selected_models:
                model = self.experts[model_name]
                pred, unc = model.predict_with_uncertainty(current_X)
                
                preds.append(pred[0])
                uncs.append(unc[0] if hasattr(unc, '__len__') else unc)
                
                # Get performance history
                if model_name in self.model_performance:
                    perf = np.mean(self.model_performance[model_name][-20:])
                else:
                    perf = 0.01
                perfs.append(perf)
                
                # Get Q-value
                q_val = self.orchestrator.q_table[self.orchestrator.get_state(regime)][
                    self.orchestrator.model_names.index(model_name)
                ]
                q_vals.append(q_val)
            
            # Aggregate predictions
            if len(preds) > 0:
                final_pred, final_unc, weights = self.aggregator.aggregate(
                    preds, uncs, perfs, q_vals
                )
            else:
                final_pred = current_y
                final_unc = 0.01
                weights = []
            
            # Update Q-learning
            reward = 1.0 / (abs(final_pred - current_y) + 1e-6)
            self.orchestrator.update(regime, selected_models, [reward] * len(selected_models))
            
            # Track performance
            for model_name in selected_models:
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = []
                model_pred = self.experts[model_name].predict(current_X)[0]
                model_error = abs(model_pred - current_y)
                self.model_performance[model_name].append(1.0 / (model_error + 1e-6))
            
            predictions.append(final_pred)
            actuals.append(current_y)
            uncertainties.append(final_unc)
            regimes_log.append(regime)
            weights_log.append(weights)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calibrate conformal predictor
        calib_preds = []
        for t in range(len(X_calib)):
            current_X = X_calib[t:t+1]
            available_models = [name for name, model in self.experts.items() if model.is_fitted]
            selected_models = self.orchestrator.select_models('Range', available_models, n_select=3)
            
            preds = []
            for model_name in selected_models:
                pred, _ = self.experts[model_name].predict_with_uncertainty(current_X)
                preds.append(pred[0])
            
            calib_preds.append(np.mean(preds))
        
        self.conformal_predictor.calibrate(np.array(calib_preds), y_calib)
        
        # Get prediction intervals
        lower, upper = self.conformal_predictor.get_prediction_intervals(predictions)
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(actuals, predictions)
        coverage = self.conformal_predictor.validate_coverage(predictions, actuals)
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'uncertainties': uncertainties,
            'lower_bound': lower,
            'upper_bound': upper,
            'regimes': regimes_log,
            'weights': weights_log,
            'metrics': metrics,
            'coverage': coverage
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("Prediction Results")
            print("=" * 60)
            print(f"RMSE: {metrics['RMSE']:.6f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
            print(f"Coverage: {coverage:.2%}")
            print("=" * 60)
        
        return results

# Example Usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_points = 500
    prices = [100.0]
    
    for t in range(1, n_points):
        ret = np.random.normal(0.0005, 0.02)
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices).reshape(-1, 1)
    
    # Initialize and train SCAF
    scaf = SCAF_Framework()
    scaf.fit(prices, verbose=True)
    
    # Make predictions
    results = scaf.predict(prices, verbose=True)
    
    # Print results
    print("\n📊 Final Metrics:")
    for key, value in results['metrics'].items():
        print(f"  {key}: {value:.4f}")