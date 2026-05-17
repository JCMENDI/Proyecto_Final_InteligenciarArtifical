import requests

ESP32_IP = "http://192.168.0.21"

def adelante():
    requests.get(f"{ESP32_IP}/adelante", timeout=0.5)

def atras():
    requests.get(f"{ESP32_IP}/atras", timeout=0.5)

def derecha():
    requests.get(f"{ESP32_IP}/derecha", timeout=0.5)

def izquierda():
    requests.get(f"{ESP32_IP}/izquierda", timeout=0.5)

def frenar():
    requests.get(f"{ESP32_IP}/frenar", timeout=0.5)