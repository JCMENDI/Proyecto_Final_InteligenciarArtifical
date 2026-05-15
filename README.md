# Proyecto Carro Autónomo con IA

Sistema de conducción autónoma para robot móvil, entrenado con una CNN sobre imágenes capturadas desde una cámara montada en el vehículo. Proyecto final del curso de Inteligencia Artificial, Universidad Rafael Landívar, 2026.

## Arquitectura (Opción B — distribuida)

```
[Celular con IP Webcam] --WiFi--> [Laptop: inferencia CNN] --WiFi--> [ESP32] --> [L298N] --> [Motores DC]
```

La cámara va montada en el carro y transmite por WiFi a la laptop. La laptop corre el modelo y le manda comandos (forward/left/right) al ESP32, que controla los motores.

## Hardware

- ESP32 (cualquier dev board)
- Driver de motores L298N
- Chasis 2WD con motores DC
- Celular Android con la app *IP Webcam* como cámara
- Powerbank + batería para motores
- Pista física (foami/cartulina mate, fondo claro, líneas oscuras)

## Estructura del repo

| Carpeta | Contenido |
|---|---|
| `firmware/` | Código Arduino del ESP32 |
| `src/` | Scripts Python (recolección, entrenamiento, inferencia) |
| `notebooks/` | Notebooks de exploración |
| `dataset/` | Imágenes etiquetadas (no en git) |
| `models/` | Pesos entrenados (no en git) |
| `docs/` | Documentación de hardware y protocolos |

## Setup

```bash
git clone <url-del-repo>
cd proyecto-carro-ia

python -m venv venv
source venv/bin/activate     # Linux / Mac
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Uso

### 1. Grabar dataset
*(pendiente)*

### 2. Entrenar el modelo
*(pendiente)*

### 3. Inferencia en vivo
*(pendiente)*

## Equipo
- *José Carlos Mendizábal Huertas - 1077222*

## Estado
🚧 En desarrollo — entrega martes 19 de mayo, 2026.