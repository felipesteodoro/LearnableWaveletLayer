"""
Módulo para criação de modelos de deep learning CNN-LSTM.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def create_cnn_lstm_model(
    input_shape: tuple, 
    num_classes: int = 3,
    conv_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 100,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.01,
    learning_rate: float = 0.001
) -> Model:
    """
    Cria um modelo híbrido CNN-LSTM para classificação de séries temporais.
    
    Arquitetura:
    - Conv1D -> MaxPooling1D -> Dropout
    - LSTM -> Dropout
    - Dense (softmax)
    
    Args:
        input_shape: Forma da entrada (time_steps, features)
        num_classes: Número de classes de saída
        conv_filters: Número de filtros na camada Conv1D
        kernel_size: Tamanho do kernel na convolução
        lstm_units: Número de unidades LSTM
        dropout_rate: Taxa de dropout
        l2_reg: Regularização L2
        learning_rate: Taxa de aprendizado
    
    Returns:
        Modelo Keras compilado
    """
    inputs = Input(shape=input_shape)
    
    # Camada Convolucional
    x = Conv1D(
        filters=conv_filters, 
        kernel_size=kernel_size, 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    
    # Camada LSTM
    x = LSTM(
        units=lstm_units, 
        return_sequences=False,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = Dropout(dropout_rate)(x)
    
    # Camada de Saída
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def create_model_for_tuning(params: dict, input_shape: tuple, num_classes: int = 3) -> Model:
    """
    Cria um modelo com base nos hiperparâmetros fornecidos pelo Optuna.
    
    Args:
        params: Dicionário com hiperparâmetros:
            - learning_rate: Taxa de aprendizado
            - l2_reg: Regularização L2
            - dropout_rate: Taxa de dropout
            - lstm_units: Unidades LSTM
            - conv_filters: Filtros Conv1D
            - kernel_size: Tamanho do kernel
        input_shape: Forma da entrada
        num_classes: Número de classes
    
    Returns:
        Modelo Keras compilado
    """
    return create_cnn_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=params.get('conv_filters', 64),
        kernel_size=params.get('kernel_size', 3),
        lstm_units=params.get('lstm_units', 100),
        dropout_rate=params.get('dropout_rate', 0.3),
        l2_reg=params.get('l2_reg', 0.01),
        learning_rate=params.get('learning_rate', 0.001)
    )


def get_default_hyperparameters() -> dict:
    """
    Retorna os hiperparâmetros padrão do modelo.
    
    Returns:
        Dicionário com hiperparâmetros
    """
    return {
        'learning_rate': 0.001,
        'l2_reg': 0.01,
        'dropout_rate': 0.3,
        'lstm_units': 100,
        'conv_filters': 64,
        'kernel_size': 3,
    }
