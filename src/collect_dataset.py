"""
collect_dataset.py — Grabación del corpus de audio propio.

Graba muestras de voz por comando y las guarda en:
    dataset/<COMANDO>/<COMANDO>_<timestamp>.wav

Parámetros de grabación (alineados con train.py):
    SAMPLE_RATE = 16000 Hz
    DURACION    = 1.5 s  (recortado a 1.0 s en entrenamiento)
    DEVICE      = 1      (Micrófono USB M13 — cambiar si es distinto)

Uso:
    python src/collect_dataset.py
    python src/collect_dataset.py --device 2   (otro micrófono)
    python src/collect_dataset.py --list        (ver micrófonos disponibles)
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import argparse
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
DURACION    = 1.5                  # segundos por muestra
DEVICE      = 1                    # índice del micrófono (M13 USB)

BASE_DIR    = Path(__file__).resolve().parent.parent / "dataset"
COMANDOS    = ["AVANZA", "RETROCEDE", "IZQUIERDA", "DERECHA",
               "DETENTE", "RUIDO_FONDO"]

# Crear carpetas si no existen
for cmd in COMANDOS:
    (BASE_DIR / cmd).mkdir(parents=True, exist_ok=True)


# ── Funciones ────────────────────────────────────────────────────────────────

def grabar(duracion=DURACION, device=DEVICE):
    """Graba <duracion> segundos de audio mono a 16 kHz."""
    audio = sd.rec(
        int(duracion * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device
    )
    sd.wait()
    return audio.flatten()


def contar_muestras():
    """Muestra cuántas muestras hay por clase."""
    print("\n  Muestras actuales:")
    total = 0
    for cmd in COMANDOS:
        carpeta = BASE_DIR / cmd
        n = len(list(carpeta.glob("*.wav")))
        barra = "█" * (n // 10) + f" ({n})"
        print(f"    {cmd:15s}: {barra}")
        total += n
    print(f"    {'TOTAL':15s}: {total} muestras")


def list_devices():
    """Lista micrófonos de entrada disponibles."""
    print("\nMicrófonos disponibles:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = " ← (predeterminado)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{marker}")
    print()


# ── Bucle principal ───────────────────────────────────────────────────────────

def main(device=DEVICE):
    print("╔══════════════════════════════════════════╗")
    print("║   GRABACIÓN DE DATASET — Comandos de Voz ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  Directorio de salida : {BASE_DIR}")
    print(f"  Micrófono (device)   : {device}")
    print(f"  Sample rate          : {SAMPLE_RATE} Hz")
    print(f"  Duración por muestra : {DURACION} s")
    print("\nComandos disponibles:")
    for i, c in enumerate(COMANDOS):
        carpeta = BASE_DIR / c
        n = len(list(carpeta.glob("*.wav")))
        print(f"  {i+1}. {c:15s} ({n} muestras)")
    print("  C. Ver conteo actual")
    print("  Q. Salir")

    while True:
        print("\n¿Qué comando vas a grabar? (1-6 / C / Q)")
        opcion = input(">>> ").strip().upper()

        if opcion == "Q":
            break
        elif opcion == "C":
            contar_muestras()
            continue

        try:
            indice = int(opcion) - 1
            if indice < 0 or indice >= len(COMANDOS):
                print("  ⚠️  Opción inválida")
                continue
        except ValueError:
            print("  ⚠️  Opción inválida")
            continue

        comando = COMANDOS[indice]
        print(f"\nGrabando: [{comando}]")
        print("ENTER = grabar muestra | Q = cambiar comando\n")

        while True:
            entrada = input(
                f"  [{comando}] ENTER=grabar | Q=cambiar >>> "
            ).strip().upper()

            if entrada == "Q":
                break

            print("  🎙️  Grabando...", end=" ", flush=True)
            audio = grabar(device=device)

            timestamp = int(time.time() * 1000)
            nombre    = f"{comando}_{timestamp}.wav"
            ruta      = BASE_DIR / comando / nombre
            wav.write(str(ruta), SAMPLE_RATE, audio)

            n = len(list((BASE_DIR / comando).glob("*.wav")))
            print(f"✅  Guardado — {comando}: {n} muestras")

    print("\n  Resumen final:")
    contar_muestras()
    print("\n✅ Grabación finalizada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grabación de dataset de voz")
    parser.add_argument("--device", type=int, default=DEVICE,
                        help=f"Índice del micrófono (default: {DEVICE})")
    parser.add_argument("--list", action="store_true",
                        help="Listar micrófonos disponibles y salir")
    args = parser.parse_args()

    if args.list:
        list_devices()
    else:
        main(device=args.device)