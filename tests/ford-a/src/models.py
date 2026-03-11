"""
Fábricas de modelos para Machine Learning (classificação) e Deep Learning.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple

# ML
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# DL
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM,
    BatchNormalization, GlobalAveragePooling1D,
    Bidirectional, LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ============================================================================
# ML CLASSIFIERS
# ============================================================================

def create_linear_svc(params: Optional[Dict] = None) -> CalibratedClassifierCV:
    """LinearSVC envolvido em CalibratedClassifierCV para predict_proba."""
    params = params or {}
    base = LinearSVC(
        C=params.get('C', 1.0),
        loss=params.get('loss', 'squared_hinge'),
        max_iter=params.get('max_iter', 10000),
        random_state=42,
    )
    return CalibratedClassifierCV(base, cv=3)


def create_sgd_classifier(params: Optional[Dict] = None) -> SGDClassifier:
    params = params or {}
    return SGDClassifier(
        loss=params.get('loss', 'hinge'),
        alpha=params.get('alpha', 1e-4),
        penalty=params.get('penalty', 'l2'),
        l1_ratio=params.get('l1_ratio', 0.15),
        learning_rate=params.get('learning_rate', 'optimal'),
        max_iter=params.get('max_iter', 5000),
        random_state=42,
    )


def create_logistic_regression(params: Optional[Dict] = None) -> LogisticRegression:
    params = params or {}
    return LogisticRegression(
        C=params.get('C', 1.0),
        penalty=params.get('penalty', 'l2'),
        l1_ratio=params.get('l1_ratio', 0.5),
        solver=params.get('solver', 'saga'),
        max_iter=params.get('max_iter', 10000),
        random_state=42,
    )


def create_rf_classifier(params: Optional[Dict] = None) -> RandomForestClassifier:
    params = params or {}
    return RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        min_samples_leaf=params.get('min_samples_leaf', 1),
        max_features=params.get('max_features', 'sqrt'),
        random_state=42,
        n_jobs=params.get('n_jobs', -1),
    )


def create_xgb_classifier(params: Optional[Dict] = None):
    import xgboost as xgb
    params = params or {}
    return xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=params.get('n_jobs', -1),
        verbosity=0,
    )


def create_lgbm_classifier(params: Optional[Dict] = None):
    import lightgbm as lgb
    params = params or {}
    return lgb.LGBMClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', -1),
        learning_rate=params.get('learning_rate', 0.1),
        num_leaves=params.get('num_leaves', 31),
        subsample=params.get('subsample', 0.8),
        random_state=42,
        n_jobs=params.get('n_jobs', -1),
        verbose=-1,
    )


def create_catboost_classifier(params: Optional[Dict] = None):
    from catboost import CatBoostClassifier
    params = params or {}
    return CatBoostClassifier(
        iterations=params.get('iterations', 200),
        depth=params.get('depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
        random_seed=42,
        thread_count=params.get('thread_count', 4),
        verbose=0,
    )


# ============================================================================
# MULTI-GPU
# ============================================================================

def get_distribute_strategy() -> tf.distribute.Strategy:
    """Retorna a melhor estratégia de distribuição disponível."""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f"⚡ MirroredStrategy: {strategy.num_replicas_in_sync} GPUs")
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("⚡ OneDeviceStrategy: GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("⚡ OneDeviceStrategy: CPU")
    return strategy


# ============================================================================
# DL — CLASSIFICATION (sigmoid + binary_crossentropy)
# ============================================================================

def create_cnn_model(input_shape: Tuple[int, int], params: Optional[Dict] = None) -> Model:
    """CNN 1D para classificação binária."""
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
    for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = Conv1D(f, k, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_reg), name=f'conv_{i+1}')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=p)(x)
        x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    for i, units in enumerate(dense_units):
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg),
                  name=f'dense_{i+1}')(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_Classifier')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None) -> Model:
    """LSTM para classificação binária."""
    params = params or {}
    units = params.get('units', [128, 64])
    dropout_rate = params.get('dropout_rate', 0.3)
    recurrent_dropout = params.get('recurrent_dropout', 0.0)
    l2_reg = params.get('l2_reg', 0.001)
    learning_rate = params.get('learning_rate', 0.001)

    inputs = Input(shape=input_shape)
    x = inputs
    for i, u in enumerate(units[:-1]):
        x = LSTM(u, return_sequences=True, dropout=dropout_rate,
                 recurrent_dropout=recurrent_dropout,
                 kernel_regularizer=l2(l2_reg), name=f'lstm_{i+1}')(x)
    x = LSTM(units[-1], return_sequences=False, dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout,
             kernel_regularizer=l2(l2_reg), name=f'lstm_{len(units)}')(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Classifier')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_lstm_model(input_shape: Tuple[int, int], params: Optional[Dict] = None) -> Model:
    """Híbrido CNN-LSTM para classificação binária."""
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
    for i, (f, k) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
        x = Conv1D(f, k, activation='relu', padding='same',
                   kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(dropout_rate)(x)
    for u in lstm_units[:-1]:
        x = LSTM(u, return_sequences=True, dropout=dropout_rate,
                 kernel_regularizer=l2(l2_reg))(x)
    x = LSTM(lstm_units[-1], return_sequences=False, dropout=dropout_rate,
             kernel_regularizer=l2(l2_reg))(x)
    for units in dense_units:
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Classifier')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# Transformer helpers
# ============================================================================

class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len}


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.att = MultiHeadAttention(key_dim=self.head_size,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout_rate)
        self.ffn = Sequential([Dense(self.ff_dim, activation="relu"),
                                Dropout(self.dropout_rate),
                                Dense(embed_dim)])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.do1 = Dropout(self.dropout_rate)
        self.do2 = Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn = self.do1(self.att(inputs, inputs, training=training), training=training)
        out1 = self.ln1(inputs + attn)
        ffn = self.do2(self.ffn(out1, training=training), training=training)
        return self.ln2(out1 + ffn)


class TransformerWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=500):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1)
        arg2 = (step + 1) * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": int(self.d_model.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy())}


def create_transformer_model(input_shape: Tuple[int, int], params: Optional[Dict] = None) -> Model:
    """Transformer para classificação binária."""
    params = params or {}
    head_size = params.get('head_size', 64)
    num_heads = params.get('num_heads', 4)
    ff_dim = params.get('ff_dim', 128)
    num_blocks = params.get('num_transformer_blocks', 2)
    mlp_units = params.get('mlp_units', [128, 64])
    dropout_rate = params.get('dropout_rate', 0.2)
    learning_rate = params.get('learning_rate', 0.001)
    l2_reg = params.get('l2_reg', 0.001)
    use_warmup = params.get('use_warmup', True)
    warmup_steps = params.get('warmup_steps', 500)

    embed_dim = head_size * num_heads
    inputs = Input(shape=input_shape)
    x = Dense(embed_dim, kernel_regularizer=l2(l2_reg))(inputs)
    x = SinusoidalPositionalEncoding(max_len=input_shape[0])(x)
    x = Dropout(dropout_rate)(x)
    for _ in range(num_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = Dense(units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Transformer_Classifier')
    if use_warmup:
        lr = TransformerWarmupSchedule(d_model=embed_dim, warmup_steps=warmup_steps)
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# LEARNED WAVELET MODELS
# ============================================================================

def create_learned_wavelet_cnn_model(input_shape, wavelet_config=None, cnn_params=None):
    import sys; sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    wavelet_config = wavelet_config or {}
    cnn_params = cnn_params or {}

    inputs = Input(shape=input_shape)
    wl = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 2),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    x = wl(inputs)
    for f in cnn_params.get('filters', [64, 128]):
        x = Conv1D(f, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name='LearnedWavelet_CNN')
    model.compile(optimizer=Adam(learning_rate=cnn_params.get('learning_rate', 0.001)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_learned_wavelet_lstm_model(input_shape, wavelet_config=None, lstm_params=None):
    import sys; sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    wavelet_config = wavelet_config or {}
    lstm_params = lstm_params or {}

    inputs = Input(shape=input_shape)
    wl = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 2),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    x = wl(inputs)
    units = lstm_params.get('units', [128, 64])
    for u in units[:-1]:
        x = LSTM(u, return_sequences=True, dropout=0.3)(x)
    x = LSTM(units[-1], return_sequences=False, dropout=0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name='LearnedWavelet_LSTM')
    model.compile(optimizer=Adam(learning_rate=lstm_params.get('learning_rate', 0.001)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_learned_wavelet_cnn_lstm_model(input_shape, wavelet_config=None, cnn_lstm_params=None):
    import sys; sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    wavelet_config = wavelet_config or {}
    cnn_lstm_params = cnn_lstm_params or {}

    inputs = Input(shape=input_shape)
    wl = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 2),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    x = wl(inputs)
    cnn_filters = cnn_lstm_params.get('cnn_filters', [64, 128])
    cnn_ks = cnn_lstm_params.get('cnn_kernel_sizes', [5, 3])
    lstm_units = cnn_lstm_params.get('lstm_units', [100, 50])
    dense_units = cnn_lstm_params.get('dense_units', [64])
    dropout_rate = cnn_lstm_params.get('dropout_rate', 0.3)
    l2_reg_val = cnn_lstm_params.get('l2_reg', 0.001)
    lr = cnn_lstm_params.get('learning_rate', 0.001)

    for f, k in zip(cnn_filters, cnn_ks):
        x = Conv1D(f, k, activation='relu', padding='same', kernel_regularizer=l2(l2_reg_val))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout_rate)(x)
    for u in lstm_units[:-1]:
        x = LSTM(u, return_sequences=True, dropout=dropout_rate, kernel_regularizer=l2(l2_reg_val))(x)
    x = LSTM(lstm_units[-1], return_sequences=False, dropout=dropout_rate, kernel_regularizer=l2(l2_reg_val))(x)
    for units in dense_units:
        x = Dense(units, activation='relu', kernel_regularizer=l2(l2_reg_val))(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs, outputs, name='LearnedWavelet_CNN_LSTM')
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_learned_wavelet_transformer_model(input_shape, wavelet_config=None, transformer_params=None):
    import sys; sys.path.append('../../../models')
    from LWT import LearnedWaveletDWT1D_QMF
    wavelet_config = wavelet_config or {}
    transformer_params = transformer_params or {}

    inputs = Input(shape=input_shape)
    wl = LearnedWaveletDWT1D_QMF(
        levels=wavelet_config.get('levels', 2),
        kernel_size=wavelet_config.get('kernel_size', 32),
        wavelet_net_units=wavelet_config.get('wavelet_net_units', 32),
        mode="concat",
        reg_energy=wavelet_config.get('reg_energy', 1e-2),
        reg_high_dc=wavelet_config.get('reg_high_dc', 1e-2),
        reg_smooth=wavelet_config.get('reg_smooth', 1e-3),
    )
    x = wl(inputs)

    head_size = transformer_params.get('head_size', 64)
    num_heads = transformer_params.get('num_heads', 4)
    ff_dim = transformer_params.get('ff_dim', 128)
    num_blocks = transformer_params.get('num_transformer_blocks', 2)
    dropout_rate = transformer_params.get('dropout_rate', 0.2)
    l2_reg_val = transformer_params.get('l2_reg', 0.001)
    embed_dim = head_size * num_heads

    x = Dense(embed_dim, kernel_regularizer=l2(l2_reg_val))(x)
    x = SinusoidalPositionalEncoding()(x)
    x = Dropout(dropout_rate)(x)
    for _ in range(num_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg_val))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='LearnedWavelet_Transformer')
    use_warmup = transformer_params.get('use_warmup', True)
    warmup_steps = transformer_params.get('warmup_steps', 500)
    lr = transformer_params.get('learning_rate', 0.001)
    if use_warmup:
        lr_sched = TransformerWarmupSchedule(d_model=embed_dim, warmup_steps=warmup_steps)
        optimizer = Adam(learning_rate=lr_sched, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    else:
        optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

def get_callbacks(model_path, patience_early=15, patience_lr=7, min_lr=1e-6,
                  monitor='val_loss', use_reduce_lr=True):
    callbacks = [
        EarlyStopping(monitor=monitor, patience=patience_early,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor=monitor,
                        save_best_only=True, verbose=0),
    ]
    if use_reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(monitor=monitor, factor=0.5,
                              patience=patience_lr, min_lr=min_lr, verbose=1)
        )
    return callbacks
