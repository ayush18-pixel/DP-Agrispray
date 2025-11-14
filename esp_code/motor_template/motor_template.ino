#include <WiFi.h>
#include <WebServer.h>

// ================== WiFi Credentials ==================
const char* ssid = "ABCD1";
const char* password = "12345678";

// ================== Motor B pins ==================
int enB = 26;   // PWM pin
int in3 = 14;   // IN3
int in4 = 27;   // IN4

// ================== Calibration ==================
// At 255 PWM → 200 ml / 21 sec = 9.52 ml/sec
// At 230 PWM → 200 ml / 28 sec = 7.14 ml/sec

float pwm_max = 255;
float pwm_min = 230;
float flow_max = 200.0 / 21.0; // 9.52 ml/sec
float flow_min = 200.0 / 28.0; // 7.14 ml/sec

WebServer server(80);

// ================== HTML Page ==================
const char MAIN_page[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>PWM Controller</title>
</head>
<body>
  <h2>Set Motor Parameters</h2>
  <form action="/setPWM">
    Volume (ml): <input type="number" name="ml" required><br><br>
    Time (sec): <input type="number" name="sec" required><br><br>
    <input type="submit" value="Submit">
  </form>
</body>
</html>
)rawliteral";

// ================== Function to map flow rate to PWM ==================
int flowToPWM(float flow) {
  float pwm = pwm_min + (flow - flow_min) * (pwm_max - pwm_min) / (flow_max - flow_min);
  return round(pwm);
}

// ================== Server Handlers ==================
void handleRoot() {
  server.send(200, "text/html", MAIN_page);
}

void handleSetPWM() {
  if (server.hasArg("ml") && server.hasArg("sec")) {
    float ml = server.arg("ml").toFloat();
    float sec = server.arg("sec").toFloat();

    float flow = ml / sec;  // required ml/sec
    int pwmValue = flowToPWM(flow);

    String message;

    if (pwmValue < pwm_min || pwmValue > pwm_max) {
      message = "Not valid! Required PWM is out of range (230–255).";
      analogWrite(enB, 0); // Stop motor
    } else {
      message = "Calculated PWM: " + String(pwmValue) + 
                " | Flow Rate: " + String(flow) + " ml/sec";
      analogWrite(enB, pwmValue);  // Dynamically set PWM
    }

    server.send(200, "text/plain", message);
  } else {
    server.send(400, "text/plain", "Missing parameters");
  }
}

// ================== Setup ==================
void setup() {
  Serial.begin(115200);

  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // Motor direction forward
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/setPWM", handleSetPWM);
  server.begin();
}

// ================== Loop ==================
void loop() {
  server.handleClient();
}
