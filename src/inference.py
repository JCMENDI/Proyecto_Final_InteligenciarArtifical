"""
inference.py — Pipeline de inferencia en tiempo real.

Flujo completo (< 500 ms extremo a extremo):
  1. Captura audio continuo del micrófono (sounddevice, Windows)
  2. Voice Activity Detection (VAD) por energía + zero-crossing rate
  3. Extracción de MFCC (igual que en entrenamiento)
  4. Normalización con estadísticas guardadas en entrenamiento
  5. Predicción con CommandCNN
  6. Envío del comando al ESP32 vía WiFi (esp32_client)
  7. Log de latencia por componente

Uso:
    python src/inference.py
    python src/inference.py --model lstm        (usar LSTM en su lugar)
    python src/inference.py --no-hardware       (solo predicción, sin ESP32)
    python src/inference.py --list-devices      (listar micrófonos disponibles)
"""

import sys
import time
import argparse
import threading
import queue
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from pathlib import Path
from collections import deque

# ── rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(Path(__file__).parent))

from model import (COMMANDS, NUM_CLASSES, SAMPLE_RATE, DURATION,
                   N_MFCC, N_FFT, HOP_LENGTH, TIME_FRAMES, INPUT_SHAPE)
from esp32_client import ESP32Client

tf.get_logger().setLevel("ERROR")

# ── parámetros de captura ────────────────────────────────────────────────────
CHUNK_DURATION  = 0.1          # segundos por chunk de audio capturado
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_DURATION)   # 1600 muestras
WINDOW_SAMPLES  = int(SAMPLE_RATE * DURATION)         # 16000 muestras (1 s)
OVERLAP_RATIO   = 0.5          # solapamiento de ventana deslizante

# ── parámetros VAD ───────────────────────────────────────────────────────────
VAD_ENERGY_THRESHOLD  = 0.01   # energía RMS mínima para considerar voz
VAD_ZCR_MAX           = 0.35   # zero-crossing rate máximo (filtra ruido puro)
VAD_HOLD_FRAMES       = 8      # frames de "hold" tras detectar voz
SILENCE_FRAMES_NEEDED = 5      # frames de silencio para resetear ventana

# ── confianza mínima para actuar ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.70   # el modelo debe estar ≥ 70 % seguro
# RUIDO_FONDO nunca dispara una acción física
REJECT_CLASS          = "RUIDO_FONDO"

# ── cooldown entre comandos (evita disparos repetidos) ───────────────────────
COMMAND_COOLDOWN_S    = 1.2    # segundos mínimos entre dos comandos


# ═══════════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL DE INFERENCIA
# ═══════════════════════════════════════════════════════════════════════════

class VoiceCommandInference:
    """
    Orquesta el pipeline completo de inferencia en tiempo real.
    Corre la captura de audio en un hilo separado para no bloquear
    el hilo de predicción.
    """

    def __init__(self, model_type: str = "cnn",
                 use_hardware: bool = True,
                 esp32_ip: str = "192.168.1.100",
                 device_index: int | None = None):

        self.model_type   = model_type.lower()
        self.use_hardware = use_hardware
        self.device_index = device_index

        # Cola entre hilo de captura y hilo de predicción
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)

        # Buffer deslizante de audio crudo
        self._ring_buffer = deque(maxlen=WINDOW_SAMPLES)

        # Estado del VAD
        self._vad_hold  = 0
        self._silence_count = 0

        # Anti-rebote: timestamp del último comando enviado
        self._last_command_time = 0.0
        self._last_command      = None

        # Estadísticas de latencia
        self._latency_log: list = []

        # ── cargar estadísticas de normalización ───────────────────────────
        norm_path = MODELS_DIR / "normalization_stats.npy"
        if norm_path.exists():
            stats = np.load(str(norm_path))
            self._norm_mean = float(stats[0])
            self._norm_std  = float(stats[1])
        else:
            print("⚠️  normalization_stats.npy no encontrado. "
                  "Usando mean=0, std=1.")
            self._norm_mean = 0.0
            self._norm_std  = 1.0

        # ── cargar modelo ──────────────────────────────────────────────────
        self.model = self._load_model()

        # ── cliente ESP32 ──────────────────────────────────────────────────
        self.esp32 = None
        if use_hardware:
            self.esp32 = ESP32Client(esp32_ip)

    # ── carga del modelo ──────────────────────────────────────────────────

    def _load_model(self) -> tf.keras.Model:
        """Carga el modelo .keras correspondiente desde models/."""
        name_map = {"cnn": "command_cnn.keras", "lstm": "command_lstm.keras"}
        filename = name_map.get(self.model_type, "command_cnn.keras")
        path = MODELS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {path}\n"
                f"Ejecuta primero: python src/train.py"
            )

        print(f"📦 Cargando modelo: {path.name} ...", end=" ", flush=True)
        model = tf.keras.models.load_model(str(path))
        print("✅")
        return model

    # ── Voice Activity Detection ──────────────────────────────────────────

    @staticmethod
    def _vad(audio_chunk: np.ndarray) -> bool:
        """
        Detecta si un chunk de audio contiene voz.

        Criterios combinados:
          1. Energía RMS > umbral  (la señal tiene amplitud suficiente)
          2. Zero-Crossing Rate < umbral máximo  (no es solo ruido de alta
             frecuencia como estática o ventilador)

        Este enfoque es más robusto que solo energía porque filtra
        ruidos de alta frecuencia que podrían activar el modelo.
        """
        if len(audio_chunk) == 0:
            return False

        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        zcr = float(np.mean(
            np.abs(np.diff(np.sign(audio_chunk))) / 2
        ))

        has_energy = rms  > VAD_ENERGY_THRESHOLD
        not_noise  = zcr  < VAD_ZCR_MAX
        return has_energy and not_noise

    # ── extracción de features ────────────────────────────────────────────

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae MFCC, normaliza y da forma al tensor de entrada del modelo.
        La normalización usa la MISMA media y std calculadas durante
        el entrenamiento (guardadas en normalization_stats.npy).
        """
        # Recortar / rellenar a 1 segundo exacto
        target = WINDOW_SAMPLES
        if len(audio) > target:
            audio = audio[:target]
        elif len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)), mode="constant")

        # Normalización de amplitud
        mx = np.max(np.abs(audio))
        if mx > 0:
            audio = audio / mx

        # MFCC — idéntico al pipeline de entrenamiento
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window="hann"
        )

        # Ajustar dimensión temporal
        if mfcc.shape[1] < TIME_FRAMES:
            mfcc = np.pad(mfcc,
                          ((0, 0), (0, TIME_FRAMES - mfcc.shape[1])),
                          mode="constant")
        else:
            mfcc = mfcc[:, :TIME_FRAMES]

        # Normalización global (media/std del dataset de entrenamiento)
        mfcc = (mfcc - self._norm_mean) / (self._norm_std + 1e-8)

        # Forma para CNN : (1, 40, 101, 1)
        if self.model_type == "cnn":
            return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
        # Forma para LSTM: (1, 101, 40)
        return mfcc.T[np.newaxis].astype(np.float32)

    # ── predicción ────────────────────────────────────────────────────────

    def _predict(self, audio: np.ndarray) -> tuple[str, float, dict]:
        """
        Ejecuta la inferencia y devuelve:
          (comando_predicho, confianza, latencias_por_componente)
        """
        latencies = {}

        # 1. Extracción de features
        t0 = time.perf_counter()
        features = self._extract_features(audio)
        latencies["features_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Inferencia del modelo
        t1 = time.perf_counter()
        proba = self.model.predict(features, verbose=0)[0]
        latencies["inference_ms"] = (time.perf_counter() - t1) * 1000

        idx        = int(np.argmax(proba))
        confidence = float(proba[idx])
        command    = COMMANDS[idx]

        return command, confidence, latencies

    # ── acción tras predicción ────────────────────────────────────────────

    def _act(self, command: str, confidence: float, latencies: dict):
        """
        Decide si enviar el comando al ESP32 y lo envía si procede.
        Condiciones para actuar:
          - Confianza ≥ CONFIDENCE_THRESHOLD
          - No es RUIDO_FONDO
          - Han pasado ≥ COMMAND_COOLDOWN_S desde el último comando
        """
        now = time.perf_counter()

        # Rechazar clase de fondo
        if command == REJECT_CLASS:
            print(f"  🔇 RUIDO_FONDO detectado ({confidence*100:.1f}%) — ignorado")
            return

        # Umbral de confianza
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"  ❓ Confianza baja: {command} ({confidence*100:.1f}%) "
                  f"— por debajo del umbral ({CONFIDENCE_THRESHOLD*100:.0f}%)")
            return

        # Cooldown anti-rebote
        elapsed = now - self._last_command_time
        if elapsed < COMMAND_COOLDOWN_S:
            print(f"  ⏳ Cooldown activo ({elapsed:.2f}s / "
                  f"{COMMAND_COOLDOWN_S}s) — ignorado")
            return

        # ── ACTUAR ────────────────────────────────────────────────────────
        t_act = time.perf_counter()
        print(f"\n  🎤 COMANDO: {command:12s} | "
              f"Confianza: {confidence*100:.1f}% | "
              f"Features: {latencies['features_ms']:.1f}ms | "
              f"Inferencia: {latencies['inference_ms']:.1f}ms", end="")

        if self.use_hardware and self.esp32:
            ok = self.esp32.send_command(command)
            hw_ms = (time.perf_counter() - t_act) * 1000
            latencies["hardware_ms"] = hw_ms
            status = "✅" if ok else "❌"
            print(f" | Hardware: {hw_ms:.1f}ms {status}")
        else:
            print(" | [sin hardware]")

        # Latencia total
        total = sum(latencies.values())
        latencies["total_ms"] = total
        self._latency_log.append(latencies.copy())
        print(f"  ⏱️  Latencia total: {total:.1f} ms "
              f"{'✅' if total < 500 else '⚠️ >500ms'}")

        self._last_command_time = now
        self._last_command      = command

    # ── hilo de captura de audio ──────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback de sounddevice — se llama cada CHUNK_DURATION segundos.
        Solo encola el chunk, NO procesa (el procesamiento va en el
        hilo principal para no bloquear el stream de audio).
        """
        if status:
            print(f"⚠️  sounddevice status: {status}")
        chunk = indata[:, 0].copy()   # mono
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass   # descarte si la cola está llena (procesamiento lento)

    # ── bucle principal ───────────────────────────────────────────────────

    def run(self):
        """Inicia el pipeline de inferencia en tiempo real."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║  ASISTENTE ROBÓTICO — Inferencia en Tiempo Real         ║")
        print("║  Universidad Rafael Landívar                            ║")
        print("╚" + "═" * 58 + "╝")
        print(f"\n  Modelo       : {self.model_type.upper()}")
        print(f"  Hardware     : {'ESP32 conectado' if self.use_hardware else 'desactivado'}")
        print(f"  Confianza min: {CONFIDENCE_THRESHOLD*100:.0f} %")
        print(f"  Cooldown     : {COMMAND_COOLDOWN_S} s")
        print(f"\n  Comandos activos: {', '.join(COMMANDS)}")
        print("\n  Escuchando... (Ctrl+C para detener)\n")

        vad_hold   = 0
        silence_ct = 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device_index,
            callback=self._audio_callback
        ):
            while True:
                try:
                    chunk = self._audio_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                # Acumular en buffer circular
                self._ring_buffer.extend(chunk.tolist())

                # VAD sobre el chunk actual
                is_voice = self._vad(chunk)

                if is_voice:
                    vad_hold   = VAD_HOLD_FRAMES
                    silence_ct = 0
                    sys.stdout.write("█")
                    sys.stdout.flush()
                else:
                    if vad_hold > 0:
                        vad_hold  -= 1
                        silence_ct = 0
                        sys.stdout.write("▒")
                        sys.stdout.flush()
                    else:
                        silence_ct += 1
                        sys.stdout.write("·")
                        sys.stdout.flush()

                # Disparar predicción cuando hay suficiente audio
                # con voz y luego silencio (fin del comando)
                if (silence_ct >= SILENCE_FRAMES_NEEDED and
                        len(self._ring_buffer) >= WINDOW_SAMPLES):

                    audio_window = np.array(
                        list(self._ring_buffer)[-WINDOW_SAMPLES:],
                        dtype=np.float32
                    )

                    # Solo predecir si la ventana tiene energía real
                    if np.sqrt(np.mean(audio_window ** 2)) > VAD_ENERGY_THRESHOLD * 0.5:
                        print()   # nueva línea tras los indicadores VAD
                        command, conf, lats = self._predict(audio_window)
                        self._act(command, conf, lats)

                    # Resetear buffer para el siguiente comando
                    self._ring_buffer.clear()
                    silence_ct = 0
                    vad_hold   = 0

    # ── reporte de latencia ───────────────────────────────────────────────

    def print_latency_report(self):
        """Imprime estadísticas de latencia al terminar."""
        if not self._latency_log:
            return

        print("\n" + "=" * 60)
        print("REPORTE DE LATENCIA")
        print("=" * 60)
        keys = ["features_ms", "inference_ms", "hardware_ms", "total_ms"]
        labels = {
            "features_ms"  : "Extracción MFCC",
            "inference_ms" : "Inferencia modelo",
            "hardware_ms"  : "Envío a hardware",
            "total_ms"     : "TOTAL"
        }
        for k in keys:
            vals = [e[k] for e in self._latency_log if k in e]
            if vals:
                print(f"  {labels[k]:22s}: "
                      f"media={np.mean(vals):.1f}ms  "
                      f"max={np.max(vals):.1f}ms  "
                      f"min={np.min(vals):.1f}ms")


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES CLI
# ═══════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """Lista todos los dispositivos de entrada de audio disponibles."""
    print("\nDispositivos de audio disponibles:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " ← (predeterminado)" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {d['name']}{marker}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRADA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Asistente Robótico — Inferencia en tiempo real"
    )
    parser.add_argument("--model", choices=["cnn", "lstm"], default="cnn",
                        help="Modelo a usar: cnn (default) o lstm")
    parser.add_argument("--no-hardware", action="store_true",
                        help="Ejecutar sin enviar comandos al ESP32")
    parser.add_argument("--esp32-ip", default="192.168.1.100",
                        help="IP del ESP32 en la red WiFi local")
    parser.add_argument("--device", type=int, default=None,
                        help="Índice del micrófono (ver --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Listar micrófonos disponibles y salir")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    engine = VoiceCommandInference(
        model_type   = args.model,
        use_hardware = not args.no_hardware,
        esp32_ip     = args.esp32_ip,
        device_index = args.device
    )

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Inferencia detenida por el usuario.")
        engine.print_latency_report()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()