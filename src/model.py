"""
model.py — Arquitecturas de modelos para clasificación de comandos de voz.

Contiene:
  - CommandCNN   : Modelo base  (CNN 2D sobre MFCC / Mel-Spectrogram)
  - CommandLSTM  : Modelo avanzado (LSTM / GRU para comandos compuestos)
  - build_model  : Función de conveniencia que devuelve el modelo solicitado
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers


# ---------------------------------------------------------------------------
# Constantes globales del proyecto
# ---------------------------------------------------------------------------
COMMANDS       = ["AVANZA", "RETROCEDE", "IZQUIERDA", "DERECHA", "DETENTE", "RUIDO_FONDO"]
NUM_CLASSES    = len(COMMANDS)          # 6
SAMPLE_RATE    = 16000                  # Hz — requerimiento mínimo del enunciado
DURATION       = 1.0                    # segundos por muestra
N_MFCC         = 40                     # coeficientes MFCC (>= 13 como pide el enunciado)
N_FFT          = 512                    # tamaño de ventana FFT
HOP_LENGTH     = 160                    # desplazamiento entre ventanas  (10 ms a 16 kHz)
N_MELS         = 64                     # bandas mel para el espectrograma
# Dimensiones de la entrada al modelo (alto x ancho x canales)
# alto  = N_MFCC    => 40 coeficientes
# ancho = frames    => ceil(SAMPLE_RATE * DURATION / HOP_LENGTH) ≈ 100 frames
TIME_FRAMES    = 1 + int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))   # ~101
INPUT_SHAPE    = (N_MFCC, TIME_FRAMES, 1)   # (40, 101, 1)


# ---------------------------------------------------------------------------
# Modelo Base — CNN 2D
# ---------------------------------------------------------------------------
def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
              dropout_rate=0.4, l2_lambda=1e-4) -> keras.Model:
    """
    CNN 2D que recibe un mapa de características MFCC de forma
    (N_MFCC, TIME_FRAMES, 1) y devuelve logits de tamaño num_classes.

    Arquitectura:
        Bloque 1 : Conv2D(32, 3x3) → BN → ReLU → Conv2D(32, 3x3) → BN → ReLU
                   → MaxPool(2x2) → Dropout(0.25)
        Bloque 2 : Conv2D(64, 3x3) → BN → ReLU → Conv2D(64, 3x3) → BN → ReLU
                   → MaxPool(2x2) → Dropout(0.25)
        Bloque 3 : Conv2D(128,3x3) → BN → ReLU → GlobalAveragePool
        Cabeza   : Dense(256) → BN → ReLU → Dropout(0.4) → Dense(num_classes)

    La regularización L2 y BatchNormalization ayudan a evitar overfitting
    con el tamaño de dataset propio (~1500 muestras).
    """
    reg = regularizers.l2(l2_lambda)

    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    # ---- Bloque convolucional 1 ----
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1a")(inputs)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ---- Bloque convolucional 2 ----
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ---- Bloque convolucional 3 ----
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)

    # GlobalAveragePooling reduce cada mapa de características a un escalar
    # → menos parámetros que Flatten, menor riesgo de overfitting
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ---- Cabeza de clasificación ----
    x = layers.Dense(256, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CommandCNN")
    return model


# ---------------------------------------------------------------------------
# Modelo Avanzado — LSTM / GRU para comandos compuestos
# ---------------------------------------------------------------------------
def build_lstm(time_steps=TIME_FRAMES, n_features=N_MFCC,
               num_classes=NUM_CLASSES, use_gru=False,
               dropout_rate=0.3) -> keras.Model:
    """
    Red recurrente (LSTM o GRU) para reconocer secuencias de palabras
    (comandos compuestos como "AVANZA RÁPIDO" o "GIRA IZQUIERDA DETENTE").

    Entrada : secuencia temporal de forma (TIME_FRAMES, N_MFCC)
              cada paso de tiempo es un vector de coeficientes MFCC
    Salida  : distribución de probabilidad sobre num_classes comandos

    Arquitectura:
        Masking → Bidireccional(LSTM/GRU 128) → Dropout
               → Bidireccional(LSTM/GRU 64)  → Dropout
               → Dense(128) → ReLU → Dropout → Dense(num_classes)

    Se usa capa Bidireccional para capturar contexto forward y backward
    en la señal de audio, lo que mejora el reconocimiento de comandos
    con varios fonemas.
    """
    RNN = layers.GRU if use_gru else layers.LSTM
    rnn_name = "GRU" if use_gru else "LSTM"

    inputs = keras.Input(shape=(time_steps, n_features),
                         name="sequence_input")

    # Masking ignora pasos de tiempo rellenos con ceros (padding)
    x = layers.Masking(mask_value=0.0, name="masking")(inputs)

    # Primera capa recurrente — devuelve secuencia completa
    x = layers.Bidirectional(
        RNN(128, return_sequences=True, dropout=0.2,
            recurrent_dropout=0.1, name=f"{rnn_name}_1"),
        name=f"bi_{rnn_name}_1"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_rnn1")(x)

    # Segunda capa recurrente — devuelve solo el último estado
    x = layers.Bidirectional(
        RNN(64, return_sequences=False, dropout=0.2,
            recurrent_dropout=0.1, name=f"{rnn_name}_2"),
        name=f"bi_{rnn_name}_2"
    )(x)
    x = layers.Dropout(dropout_rate, name="drop_rnn2")(x)

    # Cabeza de clasificación
    x = layers.Dense(128, activation="relu", name="fc_rnn")(x)
    x = layers.Dropout(dropout_rate, name="drop_fc_rnn")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model_name = f"Command{'GRU' if use_gru else 'LSTM'}"
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ---------------------------------------------------------------------------
# Función de conveniencia
# ---------------------------------------------------------------------------
def build_model(model_type: str = "cnn", **kwargs) -> keras.Model:
    """
    Construye y retorna el modelo indicado por model_type.

    Parámetros
    ----------
    model_type : "cnn"  → CommandCNN  (modelo base)
                 "lstm" → CommandLSTM (modelo avanzado, LSTM)
                 "gru"  → CommandGRU  (modelo avanzado, GRU)
    **kwargs   : argumentos adicionales para la función build_* correspondiente

    Ejemplo
    -------
    >>> model = build_model("cnn")
    >>> model.summary()
    """
    model_type = model_type.lower()
    if model_type == "cnn":
        return build_cnn(**kwargs)
    elif model_type == "lstm":
        return build_lstm(use_gru=False, **kwargs)
    elif model_type == "gru":
        return build_lstm(use_gru=True, **kwargs)
    else:
        raise ValueError(f"model_type desconocido: '{model_type}'. "
                         f"Usa 'cnn', 'lstm' o 'gru'.")


# ---------------------------------------------------------------------------
# Verificación rápida al ejecutar el archivo directamente
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MODELO BASE — CommandCNN")
    print("=" * 60)
    cnn = build_model("cnn")
    cnn.summary()
    print(f"\nInput shape : {INPUT_SHAPE}")
    print(f"Clases      : {COMMANDS}")
    print(f"N clases    : {NUM_CLASSES}")

    print("\n" + "=" * 60)
    print("MODELO AVANZADO — CommandLSTM")
    print("=" * 60)
    lstm = build_model("lstm")
    lstm.summary()

    print("\n" + "=" * 60)
    print("MODELO AVANZADO — CommandGRU")
    print("=" * 60)
    gru = build_model("gru")
    gru.summary()