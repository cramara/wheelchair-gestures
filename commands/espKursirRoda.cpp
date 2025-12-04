#include <WiFi.h>
#include <Arduino.h>

// Konfigurasi AP
const char *ssid = "MobiAI";
const char *password = "12345678";

// Konfigurasi IP statis
IPAddress local_IP(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

WiFiServer server(80);

// Motor pins
#define pwmpin1 5
#define dir1 18
#define dir2 19
#define pwmpin2 32
#define dir3 25
#define dir4 33

// PWM configuration
#define pwmChannel1 0
#define pwmChannel2 1
#define freq 15000
#define res 8

// Motor states and speed variables
int stdir[4];
int PWM1_DutyCycle = 0;
int maxspeed = 40;
int turnspeed = 60;

// Variable to track the last direction
String lastDirection = "";

// ---------- fungsi bantu untuk eksekusi perintah motor ----------
void handleDirection(const String &arah) {
  if (arah == lastDirection) {
    // kalau sama dengan perintah sebelumnya, abaikan
    return;
  }

  Serial.print("Arah : ");
  Serial.println(arah);
  lastDirection = arah; // update arah terakhir

  if (arah == "A") { // Kiri
    while (PWM1_DutyCycle <= turnspeed) {
      stdir[0] = LOW;
      stdir[1] = LOW;
      stdir[2] = HIGH;
      stdir[3] = LOW;

      digitalWrite(dir1, stdir[0]);
      digitalWrite(dir2, stdir[1]);
      digitalWrite(dir3, stdir[2]);
      digitalWrite(dir4, stdir[3]);
      ledcWrite(pwmChannel1, PWM1_DutyCycle++);
      ledcWrite(pwmChannel2, PWM1_DutyCycle++);
      delay(10);
    }
    Serial.println("Kiri");
  }
  else if (arah == "B") { // Maju
    while (PWM1_DutyCycle <= maxspeed) {
      stdir[0] = HIGH;
      stdir[1] = LOW;
      stdir[2] = HIGH;
      stdir[3] = LOW;

      digitalWrite(dir1, stdir[0]);
      digitalWrite(dir2, stdir[1]);
      digitalWrite(dir3, stdir[2]);
      digitalWrite(dir4, stdir[3]);
      ledcWrite(pwmChannel1, PWM1_DutyCycle++);
      ledcWrite(pwmChannel2, PWM1_DutyCycle++);
      delay(10);
    }
    Serial.println("Maju");
  }
  else if (arah == "C") { // Stop
    while (PWM1_DutyCycle >= 0) {
      digitalWrite(dir1, stdir[0]);
      digitalWrite(dir2, stdir[1]);
      digitalWrite(dir3, stdir[2]);
      digitalWrite(dir4, stdir[3]);
      ledcWrite(pwmChannel1, PWM1_DutyCycle--);
      ledcWrite(pwmChannel2, PWM1_DutyCycle--);
      delay(10);
    }
    Serial.println("Stop");
  }
  else if (arah == "D") { // Mundur
    while (PWM1_DutyCycle <= turnspeed) {
      stdir[0] = LOW;
      stdir[1] = HIGH;
      stdir[2] = LOW;
      stdir[3] = HIGH;

      digitalWrite(dir1, stdir[0]);
      digitalWrite(dir2, stdir[1]);
      digitalWrite(dir3, stdir[2]);
      digitalWrite(dir4, stdir[3]);
      ledcWrite(pwmChannel1, PWM1_DutyCycle++);
      ledcWrite(pwmChannel2, PWM1_DutyCycle++);
      delay(10);
    }
    Serial.println("Mundur");
  }
  else if (arah == "E") { // Kanan
    while (PWM1_DutyCycle <= turnspeed) {
      stdir[0] = HIGH;
      stdir[1] = LOW;
      stdir[2] = LOW;
      stdir[3] = LOW;

      digitalWrite(dir1, stdir[0]);
      digitalWrite(dir2, stdir[1]);
      digitalWrite(dir3, stdir[2]);
      digitalWrite(dir4, stdir[3]);
      ledcWrite(pwmChannel1, PWM1_DutyCycle++);
      ledcWrite(pwmChannel2, PWM1_DutyCycle++);
      delay(10);
    }
    Serial.println("Kanan");
  }
}

void setup() {
  Serial.begin(115200);

  // Set static IP sebelum memulai AP
  if (!WiFi.softAPConfig(local_IP, gateway, subnet)) {
    Serial.println("Gagal mengonfigurasi IP statis!");
  }

  // Mulai AP
  WiFi.softAP(ssid, password);

  Serial.print("SSID: ");
  Serial.println(ssid);

  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());

  // Start server
  server.begin();
  Serial.println("Server started");

  // Motor pin setup
  pinMode(dir1, OUTPUT);
  pinMode(dir2, OUTPUT);
  pinMode(dir3, OUTPUT);
  pinMode(dir4, OUTPUT);

  // PWM setup
  ledcSetup(pwmChannel1, freq, res);
  ledcSetup(pwmChannel2, freq, res);
  ledcAttachPin(pwmpin1, pwmChannel1);
  ledcAttachPin(pwmpin2, pwmChannel2);
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("Client connected");

    // baca satu request (bisa HTTP atau plain socket)
    String request = client.readStringUntil('\n');
    request.trim();  // buang \r dan spasi

    String arah = "";
    bool isHttp = false;

    if (request.startsWith("GET ")) {
      // --- MODE HTTP ---
      isHttp = true;
      // contoh: "GET /A HTTP/1.1"
      int firstSpace = request.indexOf(' ');
      int secondSpace = request.indexOf(' ', firstSpace + 1);
      if (firstSpace != -1 && secondSpace != -1) {
        String path = request.substring(firstSpace + 1, secondSpace); // "/A"
        if (path.length() >= 2) {
          arah = path.substring(1, 2); // ambil karakter setelah '/'
        }
      }
    } else {
      // --- MODE SOCKET (plain) ---
      // misal dari Python: "C"
      arah = request;
    }

    if (arah.length() > 0) {
      handleDirection(arah);
    } else {
      Serial.println("Perintah tidak dikenali");
    }

    // Kirim respon
    if (isHttp) {
      // HTTP response sederhana
      client.println("HTTP/1.1 200 OK");
      client.println("Content-Type: text/plain");
      client.println("Connection: close");
      client.println();
      client.print("Command: ");
      client.println(arah);
    } else {
      // mode socket: cukup kirim OK
      client.print("OK ");
      client.println(arah);
    }

    delay(10);
    client.stop();
    Serial.println("Client disconnected");
  }
}
 