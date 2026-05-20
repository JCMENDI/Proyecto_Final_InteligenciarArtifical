# Asistente Robótico por Comandos de Voz
**Universidad Rafael Landívar — Inteligencia Artificial — Primer Semestre 2026**

Sistema de reconocimiento de comandos de voz en español que controla un robot móvil en tiempo real. El pipeline va desde la captura de audio por micrófono hasta el movimiento físico de los motores, sin usar ninguna API externa de reconocimiento de voz.

---

## Arquitectura del Sistema

```
[Micrófono] → [VAD] → [MFCC 40 coef.] → [CommandCNN] → [ESP32Client] → [ESP32 + L298N] → [Motores DC]
                                        ↘ [CommandLSTM] (comandos compuestos)
```

**Configuración B del enunciado:** el laptop ejecuta la inferencia y se comunica con el ESP32 vía WiFi (HTTP). El ESP32 controla los motores a través del driver L298N.

---

## Comandos Reconocidos

| Comando | Acción del robot |
|---|---|
| AVANZA | Ambos motores hacia adelante |
| RETROCEDE | Ambos motores hacia atrás |
| IZQUIERDA | Giro diferencial izquierda |
| DERECHA | Giro diferencial derecha |
| DETENTE | Freno inmediato |
| RUIDO_FONDO | Rechazado — sin acción |

---

## Estructura del Repositorio

```
Proyecto_Final_InteligenciarArtifical/
├── dataset/
│   ├── AVANZA/          # ≥300 muestras .wav por clase
│   ├── RETROCEDE/
│   ├── IZQUIERDA/
│   ├── DERECHA/
│   ├── DETENTE/
│   └── RUIDO_FONDO/     # ≥200 muestras de silencio/ruido
├── docs/                # Métricas, matrices de confusión, curvas
├── firmware/
│   └── esp32_motor_controller/
│       └── esp32_motor_controller.ino
├── models/              # Modelos entrenados (.keras, .h5)
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── model.py         # Arquitecturas CNN y LSTM
│   ├── train.py         # Entrenamiento completo + métricas
│   ├── inference.py     # Pipeline tiempo real
│   ├── esp32_client.py  # Comunicación WiFi con ESP32
│   └── collect_dataset.py
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python **3.9 – 3.12** (TensorFlow no soporta 3.13+)
- ESP32 DevKit V1 con firmware flasheado
- Driver L298N + chasis 2WD con motores DC
- Micrófono USB (16 kHz o superior)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd Proyecto_Final_InteligenciarArtifical

# 2. Crear entorno virtual con Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1. Grabar el dataset propio

```bash
# Ver micrófonos disponibles
python src/collect_dataset.py --list

# Iniciar grabación (device=1 para micrófono USB M13)
python src/collect_dataset.py --device 1
```

Graba mínimo 300 muestras por clase. Usa `C` para ver el conteo actual.

### 2. Entrenar los modelos

```bash
cd src
python train.py
```

El script genera automáticamente en `docs/`:
- Curvas de entrenamiento (accuracy y loss)
- Matrices de confusión
- Reporte de métricas (accuracy, precision, recall, F1)
- Comparativa CNN vs LSTM

Y en `models/`:
- `command_cnn.keras` — modelo base
- `command_lstm.keras` — modelo avanzado

### 3. Configurar el ESP32

En `firmware/esp32_motor_controller/esp32_motor_controller.ino`, editar:
```cpp
const char* ssid     = "TU_RED_WIFI";
const char* password = "TU_CONTRASEÑA";
```

Flashear con Arduino IDE. Al iniciar, el Serial Monitor muestra la IP asignada.

Actualizar la IP en `src/esp32_client.py`:
```python
ESP32_IP = "http://192.168.X.XXX"
```

### 4. Ejecutar inferencia en tiempo real

```bash
# Ver micrófonos disponibles
python src/inference.py --list-devices

# Modo completo con hardware
python src/inference.py --device 1 --esp32-ip 192.168.0.19

# Solo predicción sin robot (para pruebas)
python src/inference.py --no-hardware --device 1
```

Presiona `Ctrl+C` para detener. Al cerrar imprime el reporte de latencia.

---

## Modelos

### CommandCNN — Modelo Base

CNN 2D que recibe MFCCs de forma `(40 coeficientes × 101 frames)` como imagen.

```
Input (40, 101, 1)
  → Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → GlobalAveragePool
  → Dense(256) → BN → Dropout(0.4)
  → Dense(6, softmax)
```
**Parámetros:** ~175,000 | **Entrada:** MFCC (40×101×1)

### CommandLSTM — Modelo Avanzado

Red LSTM bidireccional para comandos compuestos de dos o más palabras.

```
Input (101 frames, 40 coef.)
  → Masking → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
  → Dense(128) → Dropout → Dense(6, softmax)
```
**Parámetros:** ~354,000 | **Entrada:** secuencia MFCC (101×40)

---

## Pipeline de Audio

| Etapa | Detalle |
|---|---|
| Captura | sounddevice, 16 kHz, mono, chunks de 100ms |
| VAD | Energía RMS + Zero-Crossing Rate |
| Normalización | Amplitud máxima = 1.0 |
| MFCC | 40 coeficientes, FFT=512, hop=160 (10ms) |
| Norm. global | Media/std del dataset de entrenamiento |
| Inferencia | Modelo CNN cargado en memoria |
| Umbral | Confianza ≥ 70% para actuar |
| Cooldown | 1.2 s entre comandos consecutivos |

---

## Data Augmentation

Se aplican 5 técnicas sobre el corpus propio:

| Técnica | Descripción |
|---|---|
| Time Shifting | Desplazamiento temporal ±20% |
| Pitch Shifting | Cambio de tono ±2 semitonos |
| Noise Injection | Ruido gaussiano (factor 0.005) |
| Time Stretching | Velocidad ±15% |
| SpecAugment | Máscaras en frecuencia y tiempo |

---

## Hardware

| Componente | Especificación |
|---|---|
| Microcontrolador | ESP32 DevKit V1 |
| Driver de motores | L298N |
| Chasis | 2WD Robot Car Kit |
| Alimentación motores | LiPo 7.4V o 6×AA |
| Alimentación ESP32 | Power bank USB |
| Micrófono | USB M13, 16 kHz |

**Pines ESP32 → L298N:** IN1=GPIO27, IN2=GPIO26, IN3=GPIO25, IN4=GPIO33

---

## División del Dataset

Siguiendo las recomendaciones de la asignatura (IA y ML — Conjuntos de Datos y Validación):

| Conjunto | Porcentaje | Uso |
|---|---|---|
| Entrenamiento | 70% | El modelo aprende patrones |
| Validación | 15% | Ajuste de hiperparámetros |
| Test | 15% | Evaluación final imparcial |

La estratificación garantiza la misma proporción de clases en los tres conjuntos.

---

## Latencia Objetivo

| Componente | Tiempo objetivo |
|---|---|
| Extracción MFCC | < 50 ms |
| Inferencia CNN | < 100 ms |
| Envío WiFi al ESP32 | < 50 ms |
| **Total extremo a extremo** | **< 500 ms** |