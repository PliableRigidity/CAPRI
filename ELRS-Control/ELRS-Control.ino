#include <Arduino.h>

// =====================================================
// AUTOBOT - ESP32-C3 + ELRS + DRV8833
// =====================================================

// ---------------------
// MOTOR PINS
// ---------------------

// Left motor
#define AIN1 3
#define AIN2 2

// Right motor
#define BIN1 4
#define BIN2 5

// ---------------------
// ELRS
// ---------------------

// ELRS receiver TX -> ESP32-C3 GPIO20
#define ELRS_RX_PIN 20

HardwareSerial ELRS(1);

#define CRSF_BAUD 420000

// CRSF frame types
#define CRSF_FRAMETYPE_RC_CHANNELS_PACKED 0x16

// Receiver -> flight controller address
#define CRSF_ADDRESS_FLIGHT_CONTROLLER 0xC8

// ---------------------
// CHANNEL CALIBRATION
// ---------------------

#define CH_MIN     174
#define CH_CENTER  992
#define CH_MAX     1811

#define DEADZONE   45

// ---------------------
// SAFETY
// ---------------------

#define FAILSAFE_TIMEOUT 250

// Number of consecutive valid RC frames required
// before allowing the bot to move.
#define ARM_FRAME_COUNT 10

uint16_t channels[16] = {0};

unsigned long lastValidRCFrame = 0;
unsigned long lastDebug = 0;

bool receiverArmed = false;
uint8_t validFrameCount = 0;

// =====================================================
// CRSF BUFFER
// =====================================================

uint8_t crsfBuffer[64];

uint8_t crsfIndex = 0;
uint8_t expectedFrameSize = 0;

// Debug counters
uint32_t goodFrames = 0;
uint32_t badCRC = 0;


// =====================================================
// MOTOR CONTROL
// =====================================================

void brakeMotors() {

  // DRV8833 active brake

  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, HIGH);

  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, HIGH);
}


// -----------------------------------------------------
// LEFT MOTOR
//
// Positive = forward
// Negative = reverse
// -----------------------------------------------------

void setLeftMotor(int speed) {

  speed = constrain(speed, -255, 255);

  if (speed == 0) {

    // Active brake
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, HIGH);
  }

  else if (speed > 0) {

    // LEFT FORWARD
    analogWrite(AIN1, speed);
    digitalWrite(AIN2, LOW);
  }

  else {

    // LEFT REVERSE
    digitalWrite(AIN1, LOW);
    analogWrite(AIN2, -speed);
  }
}


// -----------------------------------------------------
// RIGHT MOTOR
// -----------------------------------------------------

void setRightMotor(int speed) {

  speed = constrain(speed, -255, 255);

  if (speed == 0) {

    // Active brake
    digitalWrite(BIN1, HIGH);
    digitalWrite(BIN2, HIGH);
  }

  else if (speed > 0) {

    // RIGHT FORWARD
    digitalWrite(BIN1, LOW);
    analogWrite(BIN2, speed);
  }

  else {

    // RIGHT REVERSE
    analogWrite(BIN1, -speed);
    digitalWrite(BIN2, LOW);
  }
}


// =====================================================
// CRSF CRC8
// =====================================================
//
// CRSF uses CRC8 DVB-S2 polynomial 0xD5
//

uint8_t crc8DvbS2(uint8_t crc, uint8_t value) {

  crc ^= value;

  for (int i = 0; i < 8; i++) {

    if (crc & 0x80) {
      crc = (crc << 1) ^ 0xD5;
    }
    else {
      crc <<= 1;
    }
  }

  return crc;
}


uint8_t calculateCRC(uint8_t *data, uint8_t length) {

  uint8_t crc = 0;

  for (uint8_t i = 0; i < length; i++) {
    crc = crc8DvbS2(crc, data[i]);
  }

  return crc;
}


// =====================================================
// DECODE 16 CRSF CHANNELS
// =====================================================

void decodeChannels(uint8_t *data) {

  uint32_t bitBuffer = 0;
  int bitsAvailable = 0;
  int byteIndex = 0;

  for (int ch = 0; ch < 16; ch++) {

    while (bitsAvailable < 11) {

      bitBuffer |=
        ((uint32_t)data[byteIndex++]) << bitsAvailable;

      bitsAvailable += 8;
    }

    channels[ch] = bitBuffer & 0x07FF;

    bitBuffer >>= 11;
    bitsAvailable -= 11;
  }

  lastValidRCFrame = millis();
}


// =====================================================
// PROCESS COMPLETE CRSF FRAME
// =====================================================

void processCRSFFrame() {

  // Layout:
  //
  // [0] address
  // [1] length
  // [2] type
  // [3...] payload
  // [last] CRC

  uint8_t frameLength = crsfBuffer[1];

  // CRC is last byte
  uint8_t receivedCRC =
    crsfBuffer[frameLength + 1];

  // CRC covers:
  // frame type + payload
  //
  // It does NOT include address or length.

  uint8_t calculatedCRC =
    calculateCRC(
      &crsfBuffer[2],
      frameLength - 1
    );

  if (receivedCRC != calculatedCRC) {

    badCRC++;
    return;
  }

  uint8_t frameType = crsfBuffer[2];

  if (frameType == CRSF_FRAMETYPE_RC_CHANNELS_PACKED) {

    decodeChannels(&crsfBuffer[3]);

    goodFrames++;

    if (validFrameCount < ARM_FRAME_COUNT) {
      validFrameCount++;
    }
  }
}


// =====================================================
// ROBUST CRSF PARSER
// =====================================================

void readCRSF() {

  while (ELRS.available()) {

    uint8_t b = ELRS.read();

    // -------------------------------------------------
    // WAIT FOR CRSF ADDRESS
    // -------------------------------------------------

    if (crsfIndex == 0) {

      // Your receiver stream uses 0xC8
      if (b != CRSF_ADDRESS_FLIGHT_CONTROLLER) {
        continue;
      }

      crsfBuffer[0] = b;
      crsfIndex = 1;

      continue;
    }


    // -------------------------------------------------
    // READ LENGTH
    // -------------------------------------------------

    if (crsfIndex == 1) {

      // Valid CRSF lengths are small.
      if (b < 2 || b > 62) {

        crsfIndex = 0;
        expectedFrameSize = 0;

        continue;
      }

      crsfBuffer[1] = b;

      // Total bytes:
      //
      // address + length byte + frame length
      expectedFrameSize = b + 2;

      crsfIndex = 2;

      continue;
    }


    // -------------------------------------------------
    // READ REST OF FRAME
    // -------------------------------------------------

    crsfBuffer[crsfIndex++] = b;


    // -------------------------------------------------
    // COMPLETE FRAME
    // -------------------------------------------------

    if (crsfIndex >= expectedFrameSize) {

      processCRSFFrame();

      crsfIndex = 0;
      expectedFrameSize = 0;
    }


    // -------------------------------------------------
    // BUFFER PROTECTION
    // -------------------------------------------------

    if (crsfIndex >= sizeof(crsfBuffer)) {

      crsfIndex = 0;
      expectedFrameSize = 0;
    }
  }
}


// =====================================================
// CHANNEL NORMALISATION
// =====================================================

int normalizeChannel(int value) {

  // Deadzone around centre

  if (abs(value - CH_CENTER) <= DEADZONE) {
    return 0;
  }

  int output;

  // Positive side
  if (value > CH_CENTER) {

    output = map(
      value,
      CH_CENTER + DEADZONE,
      CH_MAX,
      0,
      255
    );
  }

  // Negative side
  else {

    output = map(
      value,
      CH_MIN,
      CH_CENTER - DEADZONE,
      -255,
      0
    );
  }

  return constrain(output, -255, 255);
}


// =====================================================
// CHECK WHETHER STICKS ARE CENTRED
// =====================================================

bool sticksCentered() {

  return
    abs((int)channels[0] - CH_CENTER) < 70 &&
    abs((int)channels[1] - CH_CENTER) < 70;
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  // -----------------------------
  // MOTORS FIRST
  // -----------------------------

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);

  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  brakeMotors();


  // -----------------------------
  // DEBUG SERIAL
  // -----------------------------

  Serial.begin(115200);

  delay(500);


  // -----------------------------
  // ELRS UART
  // -----------------------------

  ELRS.begin(
    CRSF_BAUD,
    SERIAL_8N1,
    ELRS_RX_PIN,
    -1
  );


  Serial.println();
  Serial.println("============================");
  Serial.println("       AUTOBOT CRSF");
  Serial.println("============================");

  Serial.println("Motors locked.");
  Serial.println("Waiting for valid ELRS...");
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop() {

  readCRSF();


  // ===================================================
  // SIGNAL FAILSAFE
  // ===================================================

  if (
    lastValidRCFrame == 0 ||
    millis() - lastValidRCFrame > FAILSAFE_TIMEOUT
  ) {

    brakeMotors();

    receiverArmed = false;
    validFrameCount = 0;

    if (millis() - lastDebug > 500) {

      Serial.print("FAILSAFE");

      Serial.print(" | good:");
      Serial.print(goodFrames);

      Serial.print(" badCRC:");
      Serial.println(badCRC);

      lastDebug = millis();
    }

    return;
  }


  // ===================================================
  // START-UP SAFETY
  // ===================================================

  if (!receiverArmed) {

    brakeMotors();

    // Require several valid frames AND centred sticks.

    if (
      validFrameCount >= ARM_FRAME_COUNT &&
      sticksCentered()
    ) {

      receiverArmed = true;

      Serial.println();
      Serial.println("*** RC CONTROL ENABLED ***");
      Serial.println();
    }

    else {

      if (millis() - lastDebug > 300) {

        Serial.print("WAITING FOR NEUTRAL");

        Serial.print(" CH1:");
        Serial.print(channels[0]);

        Serial.print(" CH2:");
        Serial.print(channels[1]);

        Serial.print(" frames:");
        Serial.println(validFrameCount);

        lastDebug = millis();
      }

      return;
    }
  }


  // ===================================================
  // RC INPUT
  // ===================================================

  int rawSteering = channels[0];
  int rawThrottle = channels[1];

  int steering =
    normalizeChannel(rawSteering);

  int throttle =
    normalizeChannel(rawThrottle);


  // ===================================================
  // DIFFERENTIAL DRIVE
  // ===================================================

  int leftSpeed =
    throttle + steering;

  int rightSpeed =
    throttle - steering;


  // Preserve ratio if mixing exceeds +/-255

  int maxMagnitude =
    max(abs(leftSpeed), abs(rightSpeed));

  if (maxMagnitude > 255) {

    leftSpeed =
      (leftSpeed * 255) / maxMagnitude;

    rightSpeed =
      (rightSpeed * 255) / maxMagnitude;
  }


  // ===================================================
  // DRIVE
  // ===================================================

  setLeftMotor(leftSpeed);
  setRightMotor(rightSpeed);


  // ===================================================
  // DEBUG
  // ===================================================

  if (millis() - lastDebug > 150) {

    Serial.print("CH1:");
    Serial.print(rawSteering);

    Serial.print(" CH2:");
    Serial.print(rawThrottle);

    Serial.print(" | steer:");
    Serial.print(steering);

    Serial.print(" throttle:");
    Serial.print(throttle);

    Serial.print(" | L:");
    Serial.print(leftSpeed);

    Serial.print(" R:");
    Serial.print(rightSpeed);

    Serial.print(" | frames:");
    Serial.print(goodFrames);

    Serial.print(" crcErr:");
    Serial.println(badCRC);

    lastDebug = millis();
  }
}