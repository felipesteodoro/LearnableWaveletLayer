"""
Fábricas de modelos para Machine Learning e Deep Learning.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM, GRU,
    Flatten, BatchNormalization, GlobalAveragePooling1D,
    Bidirectional, Concatenate, Add, LayerNormalization,
    MultiHeadAttention, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============================================================================
# MODELOS DE MACHINE LEARNING CLÁSSICOS
# ============================================================================

def create_svm_pipeline(params: Optional[Dict] = None) -> Pipeline:
    """Cria pipeline SVM com normalização."""
    params = params or {}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            kernel=params.get('kernel', 'rbf'),
            epsilon=params.get('epsilon', 0.1),
        ))
    ])


def create_random_forest(params: Optional[Dict] = None) -> RandomForestRegressor:
    """Cria modelo Random Forest."""
    params = params or {}
    return RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        random_state=42,
        n_jobs=-1,
    )


def create_xgboost(params: Optional[Dict] = None):
    """Cria modelo XGBoost."""
    import xgboost as xgb
    params = params or {}
    return xgb.XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def create_lightgbm(params: Optional[Dict] = None):
    """Cria modelo LightGBM."""
    import lightgbm as lgb
    params = params or {}
    return lgb.LGBMRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', -1),
        learning_rate=params.get('learning_rate', 0.1),
        num_leaves=params.get('num_leaves', 31),
        subsample=params.get('subsample', 0.8),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ============================================================================
# MODELOS DEEP LEARNING
# ============================================================================

def create_cnn_model(
    input_shape: Tuple[int, int],
    params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo CNN 1D para regressão.
    
    Args:
        input_shape: (sequence_length, n_features)
        params: Parâmetros do modelo
    """
    params = params or {}
    
    filters = params.get('filters', [64, 128, 256])
    kernel_sizes = params.get('kernel_sizes', [7, 5, 3])
    pool_sizes = params.get('pool_sizes', [2, 2, 2])
    dense_units = params.get('dense_units', [128, 64])
    dropout_rate = params.get('dropout_rate', 0.3)
    l2_reg = params.get('l2_reg', 0.001)
    learning_rate = params.get('learning_rate', 0.001)
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Blocos convolucionais
    for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = Conv1D(f, k, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_reg), name=f'conv_{i+1}')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=p)(x)
        x = Dropout(dropout_rate)(x)
    
    x = GlobalAveragePooling1D()(x)
    
    # Camadas densas
    for i, units in enumerate(dense_units):
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg),
                  name=f'dense_{i+1}')(x)
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Regressor')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_lstm_model(
    input_shape: Tuple[int, int],
    params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo LSTM para regressão.
    """
    params = params or {}
    
    units = params.get('units', [128, 64])
    dropout_rate = params.get('dropout_rate', 0.3)
    recurrent_dropout = params.get('recurrent_dropout', 0.2)
    l2_reg = params.get('l2_reg', 0.001)
    learning_rate = params.get('learning_rate', 0.001)
    bidirectional = params.get('bidirectional', False)
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Camadas LSTM
    for i, u in enumerate(units[:-1]):
        lstm = LSTM(u, return_sequences=True, dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2(l2_reg), name=f'lstm_{i+1}')
        if bidirectional:
            x = Bidirectional(lstm)(x)
        else:
            x = lstm(x)
    
    # Última camada LSTM
    lstm = LSTM(units[-1], return_sequences=False, dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg), name=f'lstm_{len(units)}')
    if bidirectional:
        x = Bidirectional(lstm)(x)
    else:
        x = lstm(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Regressor')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_cnn_lstm_model(
    input_shape: Tuple[int, int],
    params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo híbrido CNN-LSTM para regressão.
    """
    params = params or {}
    
    cnn_filters = params.get('cnn_filters', [64, 128])
    cnn_kernel_sizes = params.get('cnn_kernel_sizes', [5, 3])
    lstm_units = params.get('lstm_units', [100, 50])
    dense_units = params.get('dense_units', [64])
    dropout_rate = params.get('dropout_rate', 0.3)
    l2_reg = params.get('l2_reg', 0.001)
    learning_rate = params.get('learning_rate', 0.001)
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Blocos CNN
    for i, (f, k) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
        x = Conv1D(f, k, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
    
    # Camadas LSTM
    for i, u in enumerate(lstm_units[:-1]):
        x = LSTM(u, return_sequences=True, dropout=dropout_rate,
                 kernel_regularizer=l2(l2_reg))(x)
    
    x = LSTM(lstm_units[-1], return_sequences=False, dropout=dropout_rate,
             kernel_regularizer=l2(l2_reg))(x)
    
    # Camadas densas
    for units in dense_units:
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Regressor')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


class TransformerBlock(tf.keras.layers.Layer):
    """Bloco Transformer para processamento de sequências."""
    
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.att = MultiHeadAttention(
            key_dim=self.head_size, 
            num_heads=self.num_heads, 
            dropout=self.dropout_rate
        )
        self.ffn = Sequential([
            Dense(self.ff_dim, activation="relu"),
            Dropout(self.dropout_rate),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_model(
    input_shape: Tuple[int, int],
    params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo Transformer para regressão de séries temporais.
    """
    params = params or {}
    
    head_size = params.get('head_size', 64)
    num_heads = params.get('num_heads', 4)
    ff_dim = params.get('ff_dim', 128)
    num_transformer_blocks = params.get('num_transformer_blocks', 2)
    mlp_units = params.get('mlp_units', [128, 64])
    dropout_rate = params.get('dropout_rate', 0.2)
    learning_rate = params.get('learning_rate', 0.001)
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Projeção inicial
    x = Dense(head_size * num_heads)(x)
    
    # Blocos Transformer
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout_rate)(x)
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    
    # MLP head
    for units in mlp_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Transformer_Regressor')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================================================
# MODELOS COM LEARNED WAVELETS
# ============================================================================

def create_learned_wavelet_cnn_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    cnn_params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo com LearnedWaveletDWT1D_QMF seguido de CNN.
    """
    import sys
    sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    
    wavelet_config = wavelet_config or {}
    cnn_params = cnn_params or {}
    
    inputs = Input(shape=input_shape)
    
    # Learned Wavelet Layer
    wavelet_layer = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 3),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    
    x = wavelet_layer(inputs)
    
    # CNN layers
    filters = cnn_params.get('filters', [64, 128])
    for f in filters:
        x = Conv1D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LearnedWavelet_CNN')
    model.compile(
        optimizer=Adam(learning_rate=cnn_params.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_learned_wavelet_lstm_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo com LearnedWaveletDWT1D_QMF seguido de LSTM.
    """
    import sys
    sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    
    wavelet_config = wavelet_config or {}
    lstm_params = lstm_params or {}
    
    inputs = Input(shape=input_shape)
    
    # Learned Wavelet Layer
    wavelet_layer = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 3),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    
    x = wavelet_layer(inputs)
    
    # LSTM layers
    units = lstm_params.get('units', [128, 64])
    for u in units[:-1]:
        x = LSTM(u, return_sequences=True, dropout=0.3)(x)
    x = LSTM(units[-1], return_sequences=False, dropout=0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LearnedWavelet_LSTM')
    model.compile(
        optimizer=Adam(learning_rate=lstm_params.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_learned_wavelet_transformer_model(
    input_shape: Tuple[int, int],
    wavelet_config: Optional[Dict] = None,
    transformer_params: Optional[Dict] = None
) -> Model:
    """
    Cria modelo com LearnedWaveletDWT1D_QMF seguido de Transformer.
    """
    import sys
    sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    
    wavelet_config = wavelet_config or {}
    transformer_params = transformer_params or {}
    
    inputs = Input(shape=input_shape)
    
    # Learned Wavelet Layer
    wavelet_layer = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 3),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    
    x = wavelet_layer(inputs)
    
    # Transformer blocks
    head_size = transformer_params.get('head_size', 64)
    num_heads = transformer_params.get('num_heads', 4)
    ff_dim = transformer_params.get('ff_dim', 128)
    num_blocks = transformer_params.get('num_transformer_blocks', 2)
    
    x = Dense(head_size * num_heads)(x)
    
    for _ in range(num_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, 0.2)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LearnedWavelet_Transformer')
    model.compile(
        optimizer=Adam(learning_rate=transformer_params.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(
    model_path: str,
    patience_early: int = 15,
    patience_lr: int = 7,
    min_lr: float = 1e-6,
    monitor: str = 'val_loss'
) -> list:
    """Retorna lista de callbacks padrão para treinamento."""
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0
        )
    ]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

MODEL_FACTORY = {
    # ML Clássico
    'SVM': create_svm_pipeline,
    'RandomForest': create_random_forest,
    'XGBoost': create_xgboost,
    'LightGBM': create_lightgbm,
    
    # Deep Learning
    'CNN': create_cnn_model,
    'LSTM': create_lstm_model,
    'CNN_LSTM': create_cnn_lstm_model,
    'Transformer': create_transformer_model,
    
    # Learned Wavelets
    'LearnedWavelet_CNN': create_learned_wavelet_cnn_model,
    'LearnedWavelet_LSTM': create_learned_wavelet_lstm_model,
    'LearnedWavelet_Transformer': create_learned_wavelet_transformer_model,
}


def create_model(model_name: str, **kwargs):
    """
    Factory function para criar modelos.
    
    Args:
        model_name: Nome do modelo
        **kwargs: Parâmetros específicos do modelo
        
    Returns:
        Modelo criado
    """
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Modelo '{model_name}' não encontrado. "
                        f"Disponíveis: {list(MODEL_FACTORY.keys())}")
    
    return MODEL_FACTORY[model_name](**kwargs)
