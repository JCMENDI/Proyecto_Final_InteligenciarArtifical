"""
esp32_client.py — Comunicación WiFi Laptop → ESP32.

Mantiene las rutas originales del firmware (.ino) que ya están
flasheadas y funcionando:
    /adelante  /atras  /derecha  /izquierda  /frenar

La clase ESP32Client envuelve esas rutas para que inference.py
pueda llamar send_command("AVANZA") sin conocer los detalles HTTP.
"""

import requests

# ── IP del ESP32 en tu red WiFi ──────────────────────────────────────────────
ESP32_IP = "http://192.168.0.19"   # ← cámbiala si el router asigna otra

# ── Mapeo: comando del modelo → ruta del firmware ───────────────────────────
COMANDOS_MAP = {
    "AVANZA"    : "adelante",
    "RETROCEDE" : "atras",
    "IZQUIERDA" : "izquierda",
    "DERECHA"   : "derecha",
    "DETENTE"   : "frenar",
    "RUIDO_FONDO": None          # nunca se envía al robot
}

TIMEOUT = 0.5   # segundos — dentro del presupuesto de 500 ms total


# ── Funciones directas (compatibilidad con código anterior) ──────────────────

def adelante():
    try: requests.get(f"{ESP32_IP}/adelante", timeout=TIMEOUT)
    except: pass

def atras():
    try: requests.get(f"{ESP32_IP}/atras", timeout=TIMEOUT)
    except: pass

def derecha():
    try: requests.get(f"{ESP32_IP}/derecha", timeout=TIMEOUT)
    except: pass

def izquierda():
    try: requests.get(f"{ESP32_IP}/izquierda", timeout=TIMEOUT)
    except: pass

def frenar():
    try: requests.get(f"{ESP32_IP}/frenar", timeout=TIMEOUT)
    except: pass


# ── Clase requerida por inference.py ────────────────────────────────────────

class ESP32Client:
    """
    Interfaz orientada a objetos sobre el firmware existente.
    inference.py instancia esta clase y llama send_command(comando).
    """

    def __init__(self, ip: str = ESP32_IP, port: int = 80):
        # Normalizar IP — aceptar con o sin "http://"
        if not ip.startswith("http"):
            ip = f"http://{ip}"
        self.base_url = ip
        print(f"🔌 ESP32Client → {self.base_url}")
        self._connected = self.ping()

    def ping(self) -> bool:
        """
        Verifica conectividad enviando /frenar (siempre seguro).
        Retorna True si el ESP32 responde.
        """
        try:
            r = requests.get(f"{self.base_url}/frenar", timeout=TIMEOUT)
            ok = r.status_code == 200
            print(f"  {'✅' if ok else '❌'} ESP32 {'responde' if ok else 'no responde'} en {self.base_url}")
            return ok
        except Exception:
            print(f"  ❌ ESP32 no responde en {self.base_url}")
            print("     Verifica que el robot esté encendido y en la misma red.")
            return False

    def send_command(self, comando: str) -> bool:
        """
        Traduce el nombre del modelo al endpoint del firmware y
        hace el GET. Retorna True si el ESP32 respondió 200.
        """
        ruta = COMANDOS_MAP.get(comando)
        if ruta is None:
            return False   # RUIDO_FONDO u otro comando sin acción
        try:
            r = requests.get(f"{self.base_url}/{ruta}", timeout=TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self):
        """Frena el robot inmediatamente."""
        frenar()


# ── Prueba standalone ────────────────────────────────────────────────────────

def ejecutar_comando(comando: str):
    """Función de conveniencia para usar sin instanciar la clase."""
    ruta = COMANDOS_MAP.get(comando)
    if ruta:
        try: requests.get(f"{ESP32_IP}/{ruta}", timeout=TIMEOUT)
        except: pass
    print(f"Comando ejecutado: {comando}")


if __name__ == "__main__":
    client = ESP32Client()
    if client._connected:
        import time
        for cmd in ["AVANZA", "DETENTE", "IZQUIERDA", "DETENTE",
                    "DERECHA", "DETENTE", "RETROCEDE", "DETENTE"]:
            print(f"→ {cmd}")
            client.send_command(cmd)
            time.sleep(1.0)