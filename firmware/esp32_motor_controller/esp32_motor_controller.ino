/*
 * esp32_motor_controller.ino
 * ─────────────────────────────────────────────────────────────────────────
 * Firmware para ESP32 DevKit V1 — Asistente Robótico por Comandos de Voz
 * Universidad Rafael Landívar · Inteligencia Artificial 2026
 *
 * Arquitectura (Configuración B del enunciado):
 *   Laptop (Python) ──WiFi HTTP──▶ ESP32 ──GPIO──▶ L298N ──▶ Motores DC
 *
 * Conexiones L298N → ESP32 DevKit V1:
 * ┌──────────────┬────────────┬────────────────────────────┐
 * │ L298N        │ ESP32 GPIO │ Función                    │
 * ├──────────────┼────────────┼────────────────────────────┤
 * │ IN1          │ 27         │ Motor A — dirección        │
 * │ IN2          │ 26         │ Motor A — dirección        │
 * │ IN3          │ 25         │ Motor B — dirección        │
 * │ IN4          │ 33         │ Motor B — dirección        │
 * │ GND          │ GND        │ Tierra común               │
 * │ VCC (7-12V)  │ —          │ Batería de motores         │
 * └──────────────┴────────────┴────────────────────────────┘
 *
 * Endpoints HTTP disponibles:
 *   GET /adelante   → avanza
 *   GET /atras      → retrocede
 *   GET /derecha    → gira derecha
 *   GET /izquierda  → gira izquierda
 *   GET /frenar     → detiene motores
 */

#include <WiFi.h>
#include <WebServer.h>

// ─── Credenciales WiFi ──────────────────────────────────────────────────────
const char* ssid     = "TIGO-EB61";       // ← tu red
const char* password = "2NB123201056";    // ← tu contraseña

// ─── Pines L298N ────────────────────────────────────────────────────────────
#define IN1 27
#define IN2 26
#define IN3 25
#define IN4 33

WebServer server(80);

// ════════════════════════════════════════════════════════════════════════════
// FUNCIONES DE MOVIMIENTO
// ════════════════════════════════════════════════════════════════════════════

void adelante() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void atras() {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}

void derecha() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
}

void izquierda() {
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void frenar() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

// ════════════════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  frenar();   // estado inicial seguro

  // ── Conectar WiFi ────────────────────────────────────────────────────
  WiFi.begin(ssid, password);
  Serial.print("Conectando a WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Conectado!");
  Serial.print("   IP del ESP32: ");
  Serial.println(WiFi.localIP());
  Serial.println("   Copia esta IP en esp32_client.py → ESP32_IP");

  // ── Registrar rutas HTTP ─────────────────────────────────────────────
  server.on("/adelante",  []() { adelante();  server.send(200, "text/plain", "adelante");  Serial.println("[CMD] AVANZA");    });
  server.on("/atras",     []() { atras();     server.send(200, "text/plain", "atras");     Serial.println("[CMD] RETROCEDE"); });
  server.on("/derecha",   []() { derecha();   server.send(200, "text/plain", "derecha");   Serial.println("[CMD] DERECHA");   });
  server.on("/izquierda", []() { izquierda(); server.send(200, "text/plain", "izquierda"); Serial.println("[CMD] IZQUIERDA"); });
  server.on("/frenar",    []() { frenar();    server.send(200, "text/plain", "frenado");   Serial.println("[CMD] DETENTE");   });

  server.begin();
  Serial.println("🌐 Servidor HTTP listo en puerto 80");
}

// ════════════════════════════════════════════════════════════════════════════
// LOOP
// ════════════════════════════════════════════════════════════════════════════

void loop() {a
  server.handleClient();
}