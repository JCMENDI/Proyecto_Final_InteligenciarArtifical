"""
train.py — Pipeline completo de entrenamiento para el Asistente Robótico
          por Comandos de Voz.

Pasos que ejecuta este script:
  1. Carga el corpus de audio desde dataset/<CLASE>/*.wav
  2. Extrae features MFCC (40 coeficientes) de cada muestra
  3. Aplica Data Augmentation (time shifting, pitch shifting,
     inyección de ruido gaussiano, time stretching, SpecAugment)
  4. Divide en Train / Validation / Test  (70 % / 15 % / 15 %)
  5. Entrena el Modelo Base (CommandCNN)
  6. Entrena el Modelo Avanzado (CommandLSTM)
  7. Evalúa ambos modelos y guarda:
       - Curvas de entrenamiento
       - Matriz de confusión
       - Reporte de métricas (accuracy, precision, recall, F1)
       - Modelos exportados (.h5 y .keras)

Uso:
    python src/train.py
"""

import os
import time
import warnings
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras

# Importar constantes y arquitecturas del módulo model.py
from model import (build_cnn, build_lstm,
                   COMMANDS, NUM_CLASSES, SAMPLE_RATE, DURATION,
                   N_MFCC, N_FFT, HOP_LENGTH, TIME_FRAMES, INPUT_SHAPE)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent   # raíz del repo
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
DOCS_DIR    = BASE_DIR / "docs"

MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ---------------------------------------------------------------------------
BATCH_SIZE   = 32
EPOCHS_CNN   = 60
EPOCHS_LSTM  = 50
LEARNING_RATE = 1e-3
TEST_SIZE     = 0.15    # 15 % para test
VAL_SIZE      = 0.15    # 15 % para validación  → entrenamiento = 70 %
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ===========================================================================
# 1. EXTRACCIÓN DE FEATURES
# ===========================================================================

def load_audio(file_path: str) -> np.ndarray:
    """
    Carga un archivo de audio, lo resamplea a SAMPLE_RATE y lo
    normaliza/recorta/rellena a exactamente DURATION segundos.
    """
    target_len = int(SAMPLE_RATE * DURATION)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Recortar o rellenar con ceros para tener longitud fija
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    # Normalización de amplitud  (evita diferencias entre micrófonos)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio.astype(np.float32)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Extrae N_MFCC coeficientes MFCC del audio y devuelve un array
    de forma (N_MFCC, TIME_FRAMES).

    ¿Por qué MFCC?
    Los Mel-Frequency Cepstral Coefficients comprimen la información
    espectral de la señal de voz imitando la percepción auditiva humana.
    La escala Mel transforma las frecuencias lineales en una escala
    logarítmica que da más resolución a las frecuencias bajas, donde
    reside la mayor parte de la información fonética del habla.

    Hiperparámetros elegidos:
      n_mfcc    = 40    (más descriptivo que 13, sin ser excesivo)
      n_fft     = 512   (32 ms a 16 kHz — resolución frecuencial adecuada)
      hop_length = 160  (10 ms — resolución temporal estándar en ASR)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )
    # Forzar dimensión temporal fija (puede variar ±1 frame por redondeo)
    if mfcc.shape[1] < TIME_FRAMES:
        pad = TIME_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    else:
        mfcc = mfcc[:, :TIME_FRAMES]

    return mfcc.astype(np.float32)    # shape: (40, 101)


# ===========================================================================
# 2. DATA AUGMENTATION
# ===========================================================================
# El enunciado exige al menos 3 técnicas. Implementamos 5.

def augment_time_shift(audio: np.ndarray, max_shift: float = 0.2) -> np.ndarray:
    """
    Time Shifting: desplaza la señal aleatoriamente hacia adelante o atrás.
    Simula distintos momentos de inicio al hablar el comando.
    max_shift = fracción de la señal total a desplazar (20 %).
    """
    shift = int(np.random.uniform(-max_shift, max_shift) * len(audio))
    return np.roll(audio, shift).astype(np.float32)


def augment_pitch_shift(audio: np.ndarray, sr: int = SAMPLE_RATE,
                        n_steps_range: float = 2.0) -> np.ndarray:
    """
    Pitch Shifting: cambia el tono de la voz sin alterar la velocidad.
    Cubre variabilidad entre hablantes de distinto timbre/registro.
    n_steps ∈ [-2, 2] semitonos.
    """
    n_steps = np.random.uniform(-n_steps_range, n_steps_range)
    return librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=n_steps
    ).astype(np.float32)


def augment_noise_injection(audio: np.ndarray,
                             noise_factor: float = 0.005) -> np.ndarray:
    """
    Inyección de Ruido Gaussiano: agrega ruido blanco de baja amplitud.
    Mejora la robustez en entornos ruidosos (laboratorio, etc.).
    noise_factor controla la relación señal/ruido.
    """
    noise = np.random.normal(0, noise_factor, len(audio))
    return (audio + noise).astype(np.float32)


def augment_time_stretch(audio: np.ndarray,
                          rate_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """
    Time Stretching: acelera o desacelera el habla sin cambiar el tono.
    Cubre variaciones de velocidad al pronunciar el comando.
    rate ∈ [0.85, 1.15] → ±15 % de velocidad.
    """
    rate = np.random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Re-ajustar a longitud fija después del stretch
    target_len = int(SAMPLE_RATE * DURATION)
    if len(stretched) > target_len:
        stretched = stretched[:target_len]
    else:
        stretched = np.pad(stretched, (0, target_len - len(stretched)),
                           mode="constant")
    return stretched.astype(np.float32)


def augment_spec_augment(mfcc: np.ndarray,
                          freq_mask_param: int = 8,
                          time_mask_param: int = 10) -> np.ndarray:
    """
    SpecAugment: enmascara bandas de frecuencia y segmentos temporales
    en el espectrograma MFCC ya extraído.
    Técnica de Google (2019) que mejora robustez sin distorsionar el audio.

    freq_mask_param : máximo de coeficientes MFCC a enmascarar
    time_mask_param : máximo de frames temporales a enmascarar
    """
    mfcc_aug = mfcc.copy()
    n_mfcc, n_frames = mfcc_aug.shape

    # Máscara de frecuencia (bloquea filas de coeficientes MFCC)
    f = np.random.randint(0, min(freq_mask_param, n_mfcc))
    f0 = np.random.randint(0, n_mfcc - f + 1)
    mfcc_aug[f0:f0 + f, :] = 0

    # Máscara de tiempo (bloquea columnas de frames)
    t = np.random.randint(0, min(time_mask_param, n_frames))
    t0 = np.random.randint(0, n_frames - t + 1)
    mfcc_aug[:, t0:t0 + t] = 0

    return mfcc_aug.astype(np.float32)


def augment_sample(audio: np.ndarray, label: str) -> list:
    """
    Genera versiones aumentadas de una muestra de audio.
    Para RUIDO_FONDO solo aplica augmentaciones conservadoras.
    Retorna lista de tuplas (mfcc, label).
    """
    samples = []
    mfcc_orig = extract_mfcc(audio)
    samples.append((mfcc_orig, label))

    if label == "RUIDO_FONDO":
        # Solo ruido leve para fondo — no queremos distorsionar demasiado
        noisy = augment_noise_injection(audio, noise_factor=0.003)
        samples.append((extract_mfcc(noisy), label))
        return samples

    # Time Shift
    shifted = augment_time_shift(audio)
    samples.append((extract_mfcc(shifted), label))

    # Pitch Shift
    pitched = augment_pitch_shift(audio)
    samples.append((extract_mfcc(pitched), label))

    # Noise Injection
    noisy = augment_noise_injection(audio)
    samples.append((extract_mfcc(noisy), label))

    # Time Stretch
    stretched = augment_time_stretch(audio)
    samples.append((extract_mfcc(stretched), label))

    # SpecAugment sobre el MFCC original
    spec_aug = augment_spec_augment(mfcc_orig)
    samples.append((spec_aug, label))

    return samples


# ===========================================================================
# 3. CARGA DEL DATASET
# ===========================================================================

def load_dataset(augment: bool = True) -> tuple:
    """
    Recorre DATASET_DIR/<CLASE>/*.wav, extrae MFCCs y aplica augmentation.

    Retorna
    -------
    X_cnn  : np.ndarray shape (N, N_MFCC, TIME_FRAMES, 1)  para la CNN
    X_lstm : np.ndarray shape (N, TIME_FRAMES, N_MFCC)      para el LSTM
    y      : np.ndarray de enteros (etiquetas codificadas)
    labels : lista de strings con los nombres de clase en orden
    """
    print("\n" + "=" * 60)
    print("CARGANDO DATASET")
    print("=" * 60)

    X_list, y_list = [], []
    class_counts = {}

    for command in COMMANDS:
        class_dir = DATASET_DIR / command
        if not class_dir.exists():
            print(f"  ⚠️  Carpeta no encontrada: {class_dir}")
            continue

        wav_files = list(class_dir.glob("*.wav")) + \
                    list(class_dir.glob("*.WAV"))

        if not wav_files:
            print(f"  ⚠️  Sin archivos .wav en: {class_dir}")
            continue

        count = 0
        for wav_path in wav_files:
            try:
                audio = load_audio(str(wav_path))
                if augment:
                    augmented = augment_sample(audio, command)
                    for mfcc_feat, lbl in augmented:
                        X_list.append(mfcc_feat)
                        y_list.append(lbl)
                        count += 1
                else:
                    mfcc_feat = extract_mfcc(audio)
                    X_list.append(mfcc_feat)
                    y_list.append(command)
                    count += 1
            except Exception as e:
                print(f"  ⚠️  Error en {wav_path.name}: {e}")
                continue

        class_counts[command] = count
        print(f"  ✅  {command:15s} → {count:5d} muestras "
              f"({'original + augmentadas' if augment else 'originales'})")

    print(f"\n  Total muestras: {len(X_list)}")

    # Codificar etiquetas a enteros
    le = LabelEncoder()
    le.fit(COMMANDS)
    y_encoded = le.transform(y_list)

    # Convertir a arrays numpy
    X_arr = np.array(X_list, dtype=np.float32)  # (N, 40, 101)

    # Normalizar globalmente: media 0, std 1
    mean = X_arr.mean()
    std  = X_arr.std() + 1e-8
    X_arr = (X_arr - mean) / std

    # Guardar estadísticas de normalización para usarlas en inferencia
    np.save(str(MODELS_DIR / "normalization_stats.npy"),
            np.array([mean, std]))

    # Dar forma para CNN: (N, 40, 101, 1)
    X_cnn  = X_arr[..., np.newaxis]

    # Dar forma para LSTM: (N, 101, 40)  ← transpuesta
    X_lstm = X_arr.transpose(0, 2, 1)

    return X_cnn, X_lstm, y_encoded, le.classes_.tolist()


# ===========================================================================
# 4. DIVISIÓN TRAIN / VALIDATION / TEST
# ===========================================================================

def split_dataset(X_cnn, X_lstm, y):
    """
    Divide en:
      - 70 % Entrenamiento  (el modelo aprende los patrones)
      - 15 % Validación     (ajuste de hiperparámetros durante entrenamiento)
      - 15 % Test           (evaluación final, datos no vistos jamás)

    Referencia: diapositivas de la asignatura — "Conjuntos de Datos y
    Validación" (IA y ML.pdf): el test set es independiente y no se usa
    durante el entrenamiento para garantizar una estimación imparcial
    del rendimiento de generalización del modelo.
    """
    # Primer split: separa el test set (15 %)
    X_cnn_tmp, X_cnn_test, X_lstm_tmp, X_lstm_test, y_tmp, y_test = \
        train_test_split(X_cnn, X_lstm, y,
                         test_size=TEST_SIZE,
                         random_state=RANDOM_SEED,
                         stratify=y)

    # Segundo split: del 85 % restante, separa validación
    # val_size relativo = 0.15 / 0.85 ≈ 0.1765
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    X_cnn_train, X_cnn_val, X_lstm_train, X_lstm_val, y_train, y_val = \
        train_test_split(X_cnn_tmp, X_lstm_tmp, y_tmp,
                         test_size=val_relative,
                         random_state=RANDOM_SEED,
                         stratify=y_tmp)

    print("\n" + "=" * 60)
    print("DIVISIÓN DEL DATASET")
    print("=" * 60)
    print(f"  Entrenamiento : {len(y_train):5d} muestras  "
          f"({len(y_train)/len(y)*100:.1f} %)")
    print(f"  Validación    : {len(y_val):5d} muestras  "
          f"({len(y_val)/len(y)*100:.1f} %)")
    print(f"  Test          : {len(y_test):5d} muestras  "
          f"({len(y_test)/len(y)*100:.1f} %)")

    return (X_cnn_train, X_cnn_val, X_cnn_test,
            X_lstm_train, X_lstm_val, X_lstm_test,
            y_train, y_val, y_test)


# ===========================================================================
# 5. CALLBACKS DE KERAS
# ===========================================================================

def get_callbacks(model_name: str) -> list:
    """
    Callbacks usados durante el entrenamiento:
      - EarlyStopping  : detiene si val_loss no mejora en 10 épocas
      - ReduceLROnPlateau : reduce LR si val_loss se estanca
      - ModelCheckpoint: guarda el mejor modelo según val_accuracy
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        )
    ]


# ===========================================================================
# 6. ENTRENAMIENTO DEL MODELO BASE (CNN)
# ===========================================================================

def train_cnn(X_train, X_val, y_train, y_val) -> tuple:
    """
    Compila y entrena el CommandCNN.
    Retorna el modelo entrenado y el historial de entrenamiento.
    """
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELO BASE — CommandCNN")
    print("=" * 60)

    model = build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n  Parámetros totales: {model.count_params():,}")
    print(f"  Epochs máximas   : {EPOCHS_CNN}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Learning rate    : {LEARNING_RATE}\n")

    # Class weights — compensa desbalance entre clases
    # (ej. RUIDO_FONDO con menos muestras que los demás comandos)
    from sklearn.utils.class_weight import compute_class_weight
    clases = np.unique(y_train)
    pesos  = compute_class_weight("balanced", classes=clases, y=y_train)
    cw     = dict(zip(clases.tolist(), pesos.tolist()))
    print(f"  Class weights  : { {COMMANDS[k]: round(v,2) for k,v in cw.items()} }\n")

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_CNN,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("cnn"),
        class_weight=cw,
        verbose=1
    )
    elapsed = time.time() - start
    print(f"\n  ⏱️  Tiempo de entrenamiento CNN: {elapsed:.1f}s")

    return model, history


# ===========================================================================
# 7. ENTRENAMIENTO DEL MODELO AVANZADO (LSTM)
# ===========================================================================

def train_lstm(X_train, X_val, y_train, y_val) -> tuple:
    """
    Compila y entrena el CommandLSTM (Bidireccional).
    Retorna el modelo entrenado y el historial.
    """
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELO AVANZADO — CommandLSTM")
    print("=" * 60)

    model = build_lstm(
        time_steps=TIME_FRAMES,
        n_features=N_MFCC,
        num_classes=NUM_CLASSES,
        use_gru=False
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n  Parámetros totales: {model.count_params():,}")
    print(f"  Epochs máximas   : {EPOCHS_LSTM}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Learning rate    : {LEARNING_RATE}\n")

    from sklearn.utils.class_weight import compute_class_weight
    clases = np.unique(y_train)
    pesos  = compute_class_weight("balanced", classes=clases, y=y_train)
    cw     = dict(zip(clases.tolist(), pesos.tolist()))

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_LSTM,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks("lstm"),
        class_weight=cw,
        verbose=1
    )
    elapsed = time.time() - start
    print(f"\n  ⏱️  Tiempo de entrenamiento LSTM: {elapsed:.1f}s")

    return model, history


# ===========================================================================
# 8. EVALUACIÓN Y MÉTRICAS
# ===========================================================================

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """
    Evalúa el modelo sobre el test set y guarda:
      - Reporte de clasificación (accuracy, precision, recall, F1)
      - Matriz de confusión (imagen PNG)
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN — {model_name}")
    print("=" * 60)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy en Test Set: {acc * 100:.2f} %\n")

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    # Guardar reporte en texto
    report_path = DOCS_DIR / f"metrics_{model_name.lower()}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Accuracy en Test Set: {acc * 100:.4f} %\n\n")
        f.write(report)
    print(f"  📄 Reporte guardado en: {report_path}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, model_name)

    return acc, y_pred


def plot_confusion_matrix(cm, class_names, model_name):
    """Genera y guarda la imagen de la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_ylabel("Etiqueta Real", fontsize=12)
    ax.set_xlabel("Etiqueta Predicha", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = DOCS_DIR / f"confusion_matrix_{model_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Matriz de confusión guardada en: {path}")


def plot_training_history(history, model_name):
    """Genera curvas de accuracy y loss (train vs validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Curvas de Entrenamiento — {model_name}",
                 fontsize=14, fontweight="bold")

    # --- Accuracy ---
    axes[0].plot(history.history["accuracy"],
                 label="Train", color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Validación", color="#FF9800",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Loss ---
    axes[1].plot(history.history["loss"],
                 label="Train", color="#2196F3", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Validación", color="#FF9800",
                 linewidth=2, linestyle="--")
    axes[1].set_title("Loss (Categorical Crossentropy)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = DOCS_DIR / f"training_curves_{model_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Curvas de entrenamiento guardadas en: {path}")


def compare_models(acc_cnn, acc_lstm):
    """Genera gráfica comparativa de accuracy entre modelos."""
    models  = ["CommandCNN\n(Modelo Base)", "CommandLSTM\n(Modelo Avanzado)"]
    accs    = [acc_cnn * 100, acc_lstm * 100]
    colors  = ["#2196F3", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, accs, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.2f} %",
                ha="center", va="bottom",
                fontweight="bold", fontsize=12)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy en Test Set (%)", fontsize=12)
    ax.set_title("Comparativa de Modelos", fontsize=14, fontweight="bold")
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.6,
               label="Umbral mínimo (80 %)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = DOCS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Comparativa guardada en: {path}")


# ===========================================================================
# 9. EXPORTAR MODELOS
# ===========================================================================

def export_models(cnn_model, lstm_model):
    """
    Guarda los modelos en formato .keras (nativo TF2) y .h5 (compatibilidad).
    """
    print("\n" + "=" * 60)
    print("EXPORTANDO MODELOS")
    print("=" * 60)

    # Formato Keras nativo
    cnn_path  = MODELS_DIR / "command_cnn.keras"
    lstm_path = MODELS_DIR / "command_lstm.keras"
    cnn_model.save(str(cnn_path))
    lstm_model.save(str(lstm_path))
    print(f"  💾 CNN  guardada en : {cnn_path}")
    print(f"  💾 LSTM guardada en : {lstm_path}")

    # Formato .h5 para compatibilidad con versiones anteriores
    cnn_model.save(str(MODELS_DIR / "command_cnn.h5"))
    lstm_model.save(str(MODELS_DIR / "command_lstm.h5"))
    print("  💾 Formatos .h5 también exportados")


# ===========================================================================
# 10. PIPELINE PRINCIPAL
# ===========================================================================

def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  PROYECTO FINAL IA — Entrenamiento de Modelos           ║")
    print("║  Universidad Rafael Landívar — Primer Semestre 2026     ║")
    print("╚" + "═" * 58 + "╝")

    # ---- Carga y augmentation ----
    X_cnn, X_lstm, y, class_names = load_dataset(augment=True)

    # ---- División del dataset (70 / 15 / 15) ----
    (X_cnn_train, X_cnn_val, X_cnn_test,
     X_lstm_train, X_lstm_val, X_lstm_test,
     y_train, y_val, y_test) = split_dataset(X_cnn, X_lstm, y)

    # ---- Entrenamiento CNN ----
    cnn_model, cnn_history = train_cnn(
        X_cnn_train, X_cnn_val, y_train, y_val
    )
    plot_training_history(cnn_history, "CommandCNN")

    # ---- Entrenamiento LSTM ----
    lstm_model, lstm_history = train_lstm(
        X_lstm_train, X_lstm_val, y_train, y_val
    )
    plot_training_history(lstm_history, "CommandLSTM")

    # ---- Evaluación sobre test set ----
    acc_cnn,  _ = evaluate_model(cnn_model,  X_cnn_test,
                                  y_test, class_names, "CommandCNN")
    acc_lstm, _ = evaluate_model(lstm_model, X_lstm_test,
                                  y_test, class_names, "CommandLSTM")

    # ---- Comparativa ----
    compare_models(acc_cnn, acc_lstm)

    # ---- Exportar ----
    export_models(cnn_model, lstm_model)

    # ---- Resumen final ----
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  CommandCNN  accuracy en test : {acc_cnn  * 100:.2f} %")
    print(f"  CommandLSTM accuracy en test : {acc_lstm * 100:.2f} %")
    print(f"\n  Archivos generados en:")
    print(f"    models/  → modelos .keras y .h5")
    print(f"    docs/    → métricas, matrices de confusión, curvas")
    print("\n✅ Entrenamiento completado.\n")


if __name__ == "__main__":
    main()