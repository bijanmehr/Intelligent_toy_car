#include <Wire.h>
#include <ADXL345.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

#include <ArduinoJson.h>
DynamicJsonDocument doc(2048);
String data = "";

// IMU
ADXL345 adxl;
void ICACHE_RAM_ATTR Interrupt_enc1();
void ICACHE_RAM_ATTR Interrupt_enc2();

int AcX,AcY,AcZ;
const int encoderIn1 = 0;
const int encoderIn2 = 12;
int countEncoder1=0;
int countEncoder2=0;

//wifi
//const char* ssid     = "cabinet_router";
//const char* password = "123456789";

const char* ssid     = "Archer_2.4";
const char* password = "Br@v0MikeH0tel0123456789";

const char* APssid = "NodeMCU";
const char* APpassword = "123456789";

const char* serverName = "http://192.168.1.154:8000/api/toycar";

uint32_t timer;


HTTPClient http;


void setup_adxl(){
  adxl.powerOn();

  //set activity/ inactivity thresholds (0-255)
  adxl.setActivityThreshold(75); //62.5mg per increment
  adxl.setInactivityThreshold(75); //62.5mg per increment
  adxl.setTimeInactivity(10); // how many seconds of no activity is inactive?

  //look of activity movement on this axes - 1 == on; 0 == off 
  adxl.setActivityX(1);
  adxl.setActivityY(1);
  adxl.setActivityZ(1);

  //look of inactivity movement on this axes - 1 == on; 0 == off
  adxl.setInactivityX(1);
  adxl.setInactivityY(1);
  adxl.setInactivityZ(1);

  //look of tap movement on this axes - 1 == on; 0 == off
  adxl.setTapDetectionOnX(0);
  adxl.setTapDetectionOnY(0);
  adxl.setTapDetectionOnZ(1);

  //set values for what is a tap, and what is a double tap (0-255)
  adxl.setTapThreshold(50); //62.5mg per increment
  adxl.setTapDuration(15); //625us per increment
  adxl.setDoubleTapLatency(80); //1.25ms per increment
  adxl.setDoubleTapWindow(200); //1.25ms per increment

  //set values for what is considered freefall (0-255)
  adxl.setFreeFallThreshold(7); //(5 - 9) recommended - 62.5mg per increment
  adxl.setFreeFallDuration(45); //(20 - 70) recommended - 5ms per increment

  //setting all interrupts to take place on int pin 1
  //I had issues with int pin 2, was unable to reset it
  adxl.setInterruptMapping( ADXL345_INT_SINGLE_TAP_BIT,   ADXL345_INT1_PIN );
  adxl.setInterruptMapping( ADXL345_INT_DOUBLE_TAP_BIT,   ADXL345_INT1_PIN );
  adxl.setInterruptMapping( ADXL345_INT_FREE_FALL_BIT,    ADXL345_INT1_PIN );
  adxl.setInterruptMapping( ADXL345_INT_ACTIVITY_BIT,     ADXL345_INT1_PIN );
  adxl.setInterruptMapping( ADXL345_INT_INACTIVITY_BIT,   ADXL345_INT1_PIN );

  //register interrupt actions - 1 == on; 0 == off  
  adxl.setInterrupt( ADXL345_INT_SINGLE_TAP_BIT, 1);
  adxl.setInterrupt( ADXL345_INT_DOUBLE_TAP_BIT, 1);
  adxl.setInterrupt( ADXL345_INT_FREE_FALL_BIT,  1);
  adxl.setInterrupt( ADXL345_INT_ACTIVITY_BIT,   1);
  adxl.setInterrupt( ADXL345_INT_INACTIVITY_BIT, 1);
  }

void setup() {
  Serial.begin(115200);
  pinMode(encoderIn1, INPUT_PULLUP);
  pinMode(encoderIn2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderIn1), Interrupt_enc1, RISING);
  attachInterrupt(digitalPinToInterrupt(encoderIn2), Interrupt_enc2, RISING);

  setup_adxl();
  
 // Mengatur WiFi ----------------------------------------------------------
  Serial.println();
  Serial.print("Configuring access point...");
  
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(APssid, APpassword);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if(millis() > 10000){
      ESP.restart();
      Serial.print("restart esp");
    }
   }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("Client IP address: ");
  Serial.println(WiFi.localIP());
   
  Serial.println("AP IP address: ");
  Serial.println(WiFi.softAPIP());

ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH) {
      type = "sketch";
    } else { // U_FS
      type = "filesystem";
    }

    // NOTE: if updating FS this would be the place to unmount FS using FS.end()
    Serial.println("Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nEnd");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) {
      Serial.println("Auth Failed");
    } else if (error == OTA_BEGIN_ERROR) {
      Serial.println("Begin Failed");
    } else if (error == OTA_CONNECT_ERROR) {
      Serial.println("Connect Failed");
    } else if (error == OTA_RECEIVE_ERROR) {
      Serial.println("Receive Failed");
    } else if (error == OTA_END_ERROR) {
      Serial.println("End Failed");
    }
  });
  ArduinoOTA.begin();
  Serial.println("OTA is ready");
  
}

void loop() {
  ArduinoOTA.handle();
  timer = millis();
  adxl.readXYZ(&AcX, &AcY, &AcZ);

  doc["time"] = timer;
  doc["ac_x"] = AcX;
  doc["ac_y"] = AcY;
  doc["ac_z"] = AcZ;
  doc["encode1"] = countEncoder1;
  doc["encode2"] = countEncoder2;
  serializeJsonPretty(doc, data);
  
  http.begin(serverName);
  http.addHeader("Content-Type", "application/json");
  int httpResponseCode = http.POST(data);
  Serial.print("HTTP Response code: ");
  Serial.println(httpResponseCode);

  data = "";

}

void Interrupt_enc1() { 
   countEncoder1++;
}
void Interrupt_enc2() { 
   countEncoder2++;
}
