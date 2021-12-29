
#include <Wire.h>
#include <ADXL345.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <string.h>
#include <ArduinoJson.h>

StaticJsonDocument<1000> doc;
//DynamicJsonDocument doc(1000);
String output = "";

ADXL345 adxl;
void ICACHE_RAM_ATTR Interrupt_enc1();
void ICACHE_RAM_ATTR Interrupt_enc2();

//const char* ssid     = "cabinet_router";
//const char* password = "123456789";



const int httpPort = 8000;

//IPAddress local_IP(192, 168, 0, 150);
//IPAddress gateway(192, 168, 0, 1);
//IPAddress subnet(255, 255, 255, 0);

IPAddress local_IP(192, 168, 1, 100);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);
 
//const char* host = "192.168.0.104";
const char* host = "192.168.1.154";
const char* serverName = "http://192.168.1.154:8000/api/toycar";
String endConnection = "Connection: close\r\n";
String contentType = "Content-Type: application/json";

WiFiClient client;
    
int AcX,AcY,AcZ;
const int encoderIn1 = 0;
const int encoderIn2 = 12;
int countEncoder1=0;
int countEncoder2=0;
    
uint32_t timer;//unsigned int
String response;

String jsonEncoder(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5) {
  return "{\"time\":\"" + String(arg0) + "\", \"AcX\":\"" + String(arg1) + "\", \"AcY\":\"" + String(arg2) +
          "\", \"AcZ\":\"" + String(arg3) + "\", \"Encode1\":\"" + String(arg4) + "\", \"Encode2\":\"" + String(arg5) + "\"}";
}

//String jsonparser(int a, int b, int c, int d, int e, int f){
//  //"{\"time\":timer,\"ac_x\":AcX,\"ac_y\":AcY,\"ac_z\":AcZ,\"encode1\":countEncoder1,\"encode2\":countEncoder2}"
//  String temp = "{\"time\":"+String(a)+","+\"ac_x\":"+String(b)+","+\"ac_y\":"+String(c)+","+\"ac_z\":"+String(d)+","+\"encode1\":"+String(e)+","+\"encode2\":"+String(f)+"}";
//  return temp;
//}

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

  
  

      setup_adxl();
      pinMode(encoderIn1, INPUT_PULLUP);
      pinMode(encoderIn2, INPUT_PULLUP);

      attachInterrupt(digitalPinToInterrupt(encoderIn1), Interrupt_enc1, RISING);
      attachInterrupt(digitalPinToInterrupt(encoderIn2), Interrupt_enc2, RISING);

  
      
     
  if (!WiFi.config(local_IP, gateway, subnet)) {
    Serial.println("STA Failed to configure");
  }
  
  // Connect to Wi-Fi network with SSID and password
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  // Print local IP address and start web server
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
      
}
     

    
void loop() {

  output = "";

  doc["time"] = timer;
  doc["ac_x"] = AcX;
  doc["ac_y"] = AcY;
  doc["ac_z"] = AcZ;
  doc["encode1"] = countEncoder1;
  doc["encode2"] = countEncoder2;
  serializeJson(doc, output);
  Serial.println();
  Serial.println(output);

  
      delay(50); 

      adxl.readXYZ(&AcX, &AcY, &AcZ);
      
      response = "";
      timer = millis();



      if (!client.connect(host, httpPort)) {
        Serial.println("connection failed");
        return;
      }

      HTTPClient http;
      http.begin(serverName);
      http.addHeader("Content-Type", "application/json");
      int httpResponseCode = http.POST("{\"time\":1000,\"ac_x\":234,\"ac_y\":156,\"ac_z\":76,\"encode1\":3,\"encode2\":9}");
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
//      Serial.println(jsonparser(timer, AcX, AcY, AcZ, countEncoder1, countEncoder2));
//      Serial.println("{\"time\":timer,\"ac_x\":AcX,\"ac_y\":AcY,\"ac_z\":AcZ,\"encode1\":countEncoder1,\"encode2\":countEncoder2}");
//      client.print(String("POST ") + "/api/toycar" + " HTTP/1.1\r\n" +
//             "Host: " + host + "\r\n" +
//             endConnection +
//             contentType + "\r\n\r\n" +
//             jsonEncoder(timer, AcX, AcY, AcZ, countEncoder1, countEncoder2)
//             + "\r\n");//
             

//      String line = client.readStringUntil('\r');
//
//      if(line == "") {
//          WiFi.disconnect();
//          while (WiFi.status() == WL_CONNECTED) {
//          delay(500);
//          Serial.print(".");
//        }
//        
//      }
//      Serial.println(jsonEncoder(timer, AcX, AcY, AcZ, countEncoder1, countEncoder2));/
//        Serial.println(String("POST ") + "/api/toycar" + " HTTP/1.1\r\n" +
//             "Host: " + host + "\r\n" +
//             endConnection +
//             contentType + "\r\n\r\n" +
//             jsonEncoder(timer, AcX, AcY, AcZ, countEncoder1, countEncoder2)
//             + "\r\n");
}

void Interrupt_enc1() { 
   countEncoder1++;
}
void Interrupt_enc2() { 
   countEncoder2++;
}
