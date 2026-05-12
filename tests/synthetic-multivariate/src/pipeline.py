"""
Pipeline de experimentos automatizado.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
import optuna
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .models import create_model, get_callbacks, MODEL_FACTORY
from .feature_extraction import WaveletFeatureExtractor
from .evaluation import RegressionEvaluator, ResultsManager, CrossValidationManager

warnings.filterwarnings('ignore')


class ExperimentPipeline:
    """
    Pipeline automatizado para execução de experimentos.
    """
    
    def __init__(
        self,
        results_dir: Path,
        random_seed: int = 42
    ):
        """
        Args:
            results_dir: Diretório para salvar resultados
            random_seed: Semente para reprodutibilidade
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.results_manager = ResultsManager(self.results_dir)
        self.evaluator = RegressionEvaluator()
        self.feature_extractor = None
        
    def set_feature_extractor(self, extractor: WaveletFeatureExtractor):
        """Define o extrator de features."""
        self.feature_extractor = extractor
    
    def run_ml_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        experiment_name: str,
        param_grid: Optional[Dict] = None,
        use_wavelet_features: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Executa experimento de ML clássico.
        """
        print(f"\n{'='*60}")
        print(f"Experimento: {experiment_name} - {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Extrair features se necessário
        if use_wavelet_features and self.feature_extractor is not None:
            print("Extraindo features wavelet...")
            X_train_feat = self.feature_extractor.extract_features(X_train)
            X_test_feat = self.feature_extractor.extract_features(X_test)
        else:
            # Flatten para ML
            X_train_feat = X_train.reshape(X_train.shape[0], -1)
            X_test_feat = X_test.reshape(X_test.shape[0], -1)
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)
        
        # Criar modelo
        model = create_model(model_name)
        
        # Grid Search com TimeSeriesSplit
        if param_grid:
            print(f"Executando Grid Search com {cv_folds} folds...")
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Para sklearn pipeline, acessar o estimador
            if hasattr(model, 'named_steps'):
                estimator = model.named_steps[list(model.named_steps.keys())[-1]]
                grid_search = GridSearchCV(
                    estimator, param_grid, cv=tscv, 
                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train_scaled, y_train)
                best_params = grid_search.best_params_
                model.named_steps[list(model.named_steps.keys())[-1]] = grid_search.best_estimator_
            else:
                grid_search = GridSearchCV(
                    model, param_grid, cv=tscv,
                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train_scaled, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            
            print(f"Melhores parâmetros: {best_params}")
        else:
            best_params = {}
            model.fit(X_train_scaled, y_train)
        
        # Predições
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métricas
        train_metrics = self.evaluator.evaluate(y_train, y_pred_train, prefix='train')
        test_metrics = self.evaluator.evaluate(y_test, y_pred_test, prefix='test')
        
        elapsed_time = time.time() - start_time
        
        # Resultados
        results = {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params,
            'elapsed_time': elapsed_time,
            'feature_type': 'wavelet' if use_wavelet_features else 'raw',
        }
        
        # Salvar no log
        self.results_manager.log_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            metrics=test_metrics,
            config={
                'best_params': best_params,
                'feature_type': results['feature_type'],
            },
            additional_info={'elapsed_time': elapsed_time}
        )
        
        # Salvar predições
        self.results_manager.save_predictions(
            f"{experiment_name}_{model_name}",
            y_test, y_pred_test
        )
        
        # Salvar modelo
        self.results_manager.save_model_weights(
            f"{experiment_name}_{model_name}",
            model,
            framework='sklearn' if model_name in ['SVM', 'RandomForest'] else 
                      'xgboost' if model_name == 'XGBoost' else 'lightgbm'
        )
        
        print(f"\nResultados (Test):")
        print(f"  RMSE: {test_metrics.get('test_rmse', 'N/A'):.6f}")
        print(f"  MAE:  {test_metrics.get('test_mae', 'N/A'):.6f}")
        print(f"  R²:   {test_metrics.get('test_r2', 'N/A'):.6f}")
        print(f"Tempo: {elapsed_time:.2f}s")
        
        return results
    
    def run_dl_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        experiment_name: str,
        model_params: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        use_wavelet_features: bool = False,
        wavelet_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Executa experimento de Deep Learning.
        """
        import tensorflow as tf
        
        print(f"\n{'='*60}")
        print(f"Experimento: {experiment_name} - {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        training_config = training_config or {}
        model_params = model_params or {}
        
        # Preparar dados
        if use_wavelet_features and self.feature_extractor is not None:
            print("Transformando com wavelets fixas...")
            X_train_prep = self.feature_extractor.get_multilevel_coefficients(X_train)
            X_val_prep = self.feature_extractor.get_multilevel_coefficients(X_val)
            X_test_prep = self.feature_extractor.get_multilevel_coefficients(X_test)
        else:
            # Adicionar dimensão de canal se necessário
            if len(X_train.shape) == 2:
                X_train_prep = X_train[..., np.newaxis]
                X_val_prep = X_val[..., np.newaxis]
                X_test_prep = X_test[..., np.newaxis]
            else:
                X_train_prep = X_train
                X_val_prep = X_val
                X_test_prep = X_test
        
        input_shape = X_train_prep.shape[1:]
        print(f"Input shape: {input_shape}")
        
        # Criar modelo
        if model_name.startswith('LearnedWavelet'):
            model = create_model(
                model_name,
                input_shape=input_shape,
                wavelet_config=wavelet_config,
                **{k: v for k, v in model_params.items() if k not in ['input_shape']}
            )
        else:
            model = create_model(
                model_name,
                input_shape=input_shape,
                params=model_params
            )
        
        model.summary()
        
        # Callbacks
        model_path = str(self.results_dir / 'model_weights' / f'{experiment_name}_{model_name}.keras')
        callbacks = get_callbacks(
            model_path,
            patience_early=training_config.get('early_stopping_patience', 15),
            patience_lr=training_config.get('reduce_lr_patience', 7),
            min_lr=training_config.get('min_lr', 1e-6)
        )
        
        # Treinamento
        history = model.fit(
            X_train_prep, y_train,
            validation_data=(X_val_prep, y_val),
            epochs=training_config.get('epochs', 100),
            batch_size=training_config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=training_config.get('verbose', 1)
        )
        
        # Predições
        y_pred_train = model.predict(X_train_prep, verbose=0).flatten()
        y_pred_val = model.predict(X_val_prep, verbose=0).flatten()
        y_pred_test = model.predict(X_test_prep, verbose=0).flatten()
        
        # Métricas
        train_metrics = self.evaluator.evaluate(y_train, y_pred_train, prefix='train')
        val_metrics = self.evaluator.evaluate(y_val, y_pred_val, prefix='val')
        test_metrics = self.evaluator.evaluate(y_test, y_pred_test, prefix='test')
        
        elapsed_time = time.time() - start_time
        
        # Resultados
        results = {
            'experiment_name': experiment_name,
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
            'elapsed_time': elapsed_time,
            'epochs_trained': len(history.history['loss']),
            'feature_type': 'wavelet' if use_wavelet_features else 'raw',
        }
        
        # Salvar no log
        self.results_manager.log_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            metrics=test_metrics,
            config={
                'model_params': model_params,
                'training_config': training_config,
                'feature_type': results['feature_type'],
            },
            additional_info={
                'elapsed_time': elapsed_time,
                'epochs_trained': results['epochs_trained']
            }
        )
        
        # Salvar predições
        self.results_manager.save_predictions(
            f"{experiment_name}_{model_name}",
            y_test, y_pred_test
        )
        
        print(f"\nResultados (Test):")
        print(f"  RMSE: {test_metrics.get('test_rmse', 'N/A'):.6f}")
        print(f"  MAE:  {test_metrics.get('test_mae', 'N/A'):.6f}")
        print(f"  R²:   {test_metrics.get('test_r2', 'N/A'):.6f}")
        print(f"Tempo: {elapsed_time:.2f}s, Epochs: {results['epochs_trained']}")
        
        return results


class OptunaOptimizer:
    """
    Otimizador de hiperparâmetros usando Optuna.
    """
    
    def __init__(
        self,
        n_trials: int = 50,
        timeout: int = 3600,
        direction: str = "minimize"
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
    
    def optimize_ml_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_ranges: Dict[str, Any]
    ) -> Tuple[Dict, float]:
        """
        Otimiza hiperparâmetros de modelo ML usando Optuna.
        """
        from sklearn.metrics import mean_squared_error
        
        def objective(trial):
            params = {}
            for param_name, param_config in param_ranges.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            model = create_model(model_name, params=params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return study.best_params, study.best_value
    
    def optimize_dl_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_ranges: Dict[str, Any],
        epochs: int = 50
    ) -> Tuple[Dict, float]:
        """
        Otimiza hiperparâmetros de modelo DL usando Optuna.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        
        def objective(trial):
            params = {}
            for param_name, param_config in param_ranges.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            tf.keras.backend.clear_session()
            
            input_shape = X_train.shape[1:]
            model = create_model(model_name, input_shape=input_shape, params=params)
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
            ]
            
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
            return val_loss
        
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction=self.direction, pruner=pruner)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return study.best_params, study.best_value
