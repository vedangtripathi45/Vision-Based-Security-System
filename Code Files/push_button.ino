// Push button connected to GPIO 18
const int buttonPin = 2;

// Variable to store button state
int buttonState = 1;
int lastButtonState = 1;  // To detect changes
bool state=0;
bool auth=0;
bool prev_auth=0;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Set button pin as input with internal pull-up resistor
  pinMode(buttonPin, INPUT_PULLUP);
  
  //Serial.println("ESP32 Button Reader Ready!");
  //Serial.println("Press the button...");
}

void loop() {
  // Read button state
  buttonState = digitalRead(buttonPin);
  //Serial.println(buttonState);
  // Print when button state changes (avoids flooding serial monitor)
  if (buttonState==1 and lastButtonState==0) {
      state=1-state;
      Serial.write(state);
      
      delay(50);  // Debounce delay
  }

  if (Serial.available() > 0) {
    auth = Serial.read();   // Reads 0 or 1 from Python
    
    // You can use it to control LED, motor, flags, etc
    if (auth == 1 and prev_auth==0) {
      // DO SOMETHING WHEN 1 RECEIVED
      // e.g. turn LED ON
      Serial.print("I am ESP 32 I recieved authorization");
    }
  }
  prev_auth=auth;
  lastButtonState = buttonState;
}
