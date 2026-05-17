#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "TU_WIFI";
const char* password = "TU_CONTRASEÑA";

#define IN1 27
#define IN2 26
#define IN3 25
#define IN4 33

WebServer server(80);

void adelante()  { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW); }
void atras()     { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH); }
void derecha()   { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH); }
void izquierda() { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW); }
void frenar()    { digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);  digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW); }

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  WiFi.begin(ssid, password);
  Serial.print("Conectando a WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nConectado!");
  Serial.println(WiFi.localIP());

  server.on("/adelante",  []() { adelante();  server.send(200, "text/plain", "adelante"); });
  server.on("/atras",     []() { atras();     server.send(200, "text/plain", "atras"); });
  server.on("/derecha",   []() { derecha();   server.send(200, "text/plain", "derecha"); });
  server.on("/izquierda", []() { izquierda(); server.send(200, "text/plain", "izquierda"); });
  server.on("/frenar",    []() { frenar();    server.send(200, "text/plain", "frenado"); });

  server.begin();
}

void loop() {
  server.handleClient();
}