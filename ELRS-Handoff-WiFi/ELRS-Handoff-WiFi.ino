#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_random.h>

// =====================================================
// AUTOBOT - ESP32-C3 + ELRS + DRV8833
// RADIO / PC HANDOFF BENCH TEST  +  WI-FI PC LINK
// =====================================================
//
// Bench-test sketch. It answers three questions:
//   1. Who has control (RadioMaster sticks or the PC)?
//   2. What happens when control is handed over?
//   3. Do the motors stop on disarm, radio loss and PC loss?
//
// The radio receive path, channel calibration, mixing and wheel
// polarity are copied from ELRS-Control/ELRS-Control.ino, which is
// NOT modified. No IMU, no vision, no autonomous control here.
//
// The RadioMaster is the only authority:
//   - a switch on ARM_CHANNEL arms and disarms,
//   - a switch on MODE_CHANNEL selects RADIO or PC.
// The PC can only send drive commands. Nothing it sends can arm the
// robot, select PC mode or clear a fault.
//
// This is ELRS-Handoff-Bench.ino (the USB version tested on the robot)
// with one addition: the same PC command lines can also arrive over
// Wi-Fi. The robot is a password-protected access point; the laptop
// joins it directly and sends UDP datagrams. Wi-Fi commands go through
// exactly the same checks as USB commands. There is no second motor
// path, and only one of USB / Wi-Fi can hold the PC input at a time.
// USB does not have to be connected for anything.
//
// The USB port carries a line protocol (see README.md next to this
// file). Do not add Serial.print calls: use the lines that exist.
//
// Build: Tools > Board "ESP32C3 Dev Module", "USB CDC On Boot: Enabled".
// (That setting only chooses which port "Serial" is. Nothing waits for USB.)

// ---------------------
// MOTOR PINS (same as ELRS-Control.ino)
// ---------------------

// Left motor
#define AIN1 3
#define AIN2 2

// Right motor
#define BIN1 4
#define BIN2 5

// ---------------------
// ELRS (same as ELRS-Control.ino)
// ---------------------

// ELRS receiver TX -> ESP32-C3 GPIO20. UART1, receive only.
#define ELRS_RX_PIN 20

HardwareSerial ELRS(1);

#define CRSF_BAUD 420000

// CRSF frame types
#define CRSF_FRAMETYPE_RC_CHANNELS_PACKED 0x16
#define CRSF_FRAMETYPE_LINK_STATISTICS    0x14

// Receiver -> flight controller address
#define CRSF_ADDRESS_FLIGHT_CONTROLLER 0xC8

// CRSF length byte = type + payload + CRC.
// 16 channels x 11 bits = 22 payload bytes, so an RC frame is 24.
// The original sketch decoded 22 bytes whatever the length said.
#define CRSF_RC_FRAME_LENGTH         24
#define CRSF_LINK_STATS_MIN_LENGTH   12

// ---------------------
// CHANNEL CALIBRATION (same as ELRS-Control.ino)
// ---------------------

#define CH_MIN     174
#define CH_CENTER  992
#define CH_MAX     1811

#define DEADZONE   45

// ---------------------
// SAFETY (same values as ELRS-Control.ino)
// ---------------------

// No CRC-valid RC frame for this long = radio lost.
#define FAILSAFE_TIMEOUT 250

// Consecutive valid RC frames required before the radio link counts.
// (This qualifies the link. It does not arm anything.)
#define ARM_FRAME_COUNT 10

// ---------------------
// RADIO CHANNEL MAP   *** DEFAULTS, NOT CONFIRMED ***
// ---------------------
//
// Channel numbers as shown on the RadioMaster (CH1 = 1).
// Watch the "CH" line while disarmed, flip each switch, and change
// these four lines to match what you see.

#define STEER_CHANNEL     1   // as in ELRS-Control.ino
#define THROTTLE_CHANNEL  2   // as in ELRS-Control.ino
#define ARM_CHANNEL       5   // two-position switch: ARM / DISARM
#define MODE_CHANNEL      6   // two-position switch: RADIO / PC

// Which switch position means what. 1 = the HIGH position.
#define ARM_WHEN_HIGH  1      // HIGH = armed,  LOW = disarmed
#define PC_WHEN_HIGH   1      // HIGH = PC,     LOW = RADIO

// Switch decoding in raw CRSF units (172 = 988 us, 992 = 1500 us,
// 1811 = 2012 us). A value between the two bands (a three-position
// switch in the middle, a slider) or outside the valid range is
// INVALID. An invalid switch never grants drive.
#define SWITCH_VALID_MIN  100
#define SWITCH_LOW_MAX    600    // <= this is LOW  (about 1255 us)
#define SWITCH_HIGH_MIN   1400   // >= this is HIGH (about 1755 us)
#define SWITCH_VALID_MAX  1950

// The receiver also reports link quality (LINK_STATISTICS). A fresh
// report of 0 % is treated as radio lost, in addition to the RC frame
// timeout above. Reports are never required: no report = no opinion.
#define LINK_LOSS_ON_LQ_ZERO  1
#define LINK_STATS_MAX_AGE    1000   // ms

// ---------------------
// PC COMMANDS
// ---------------------

// Accepted PC commands older than this = PC lost. Same as the radio.
#define PC_TIMEOUT 250

// PC throttle and steer use the same -255..255 units as the stick
// values after normalizeChannel(). The firmware clamps them to this,
// whatever the PC asks for.
#define PC_COMMAND_LIMIT 128

// ---------------------
// WI-FI PC LINK
// ---------------------
//
// The robot is the access point. Join it from the laptop; the robot is
// then 192.168.4.1. CHANGE THE PASSWORD before uploading (8..63
// characters, no spaces in the name).

#define WIFI_AP_SSID      "AutoBot-Handoff"
#define WIFI_AP_PASSWORD  "autobot123"
#define WIFI_AP_CHANNEL   6      // 1, 6 or 11; the ELRS link shares this band
#define WIFI_UDP_PORT     4210

// Status lines go to the last address that sent a well-formed
// datagram. Forgotten after this long without one.
#define WIFI_PEER_TIMEOUT 3000   // ms

// Datagrams handled per millisecond, so a flood cannot hold up the
// radio or the stop checks.
#define WIFI_MAX_DATAGRAMS 8

// Some ESP32-C3 Super Mini boards only work as an access point at
// reduced transmit power, and less power is kinder to the ELRS
// receiver next to it. 1 = 8.5 dBm (enough across a bench or arena).
// Not tried on this robot: leave at 0 unless the network does not
// appear or the radio link suffers.
#ifndef WIFI_REDUCED_TX_POWER
#define WIFI_REDUCED_TX_POWER 0
#endif

// ---------------------
// DIAGNOSTICS
// ---------------------

#define STATUS_PERIOD_MS        200   // "ST" line heartbeat
#define STATUS_MIN_GAP_MS       20    // ... and no faster than this on changes
#define CHANNEL_PERIOD_IDLE_MS  250   // "CH" line while disarmed
#define CHANNEL_PERIOD_ARMED_MS 1000  // "CH" line while armed

// The ESP32-C3's own USB serial port. With "USB CDC On Boot" disabled,
// Serial would be UART0 on GPIO20/21, which is the ELRS receiver's pin.
#if !ARDUINO_USB_CDC_ON_BOOT
#error "Set Tools > USB CDC On Boot: Enabled (this sketch talks to the PC on the ESP32-C3's own USB port)"
#endif

#define USB_PORT Serial

static_assert(STEER_CHANNEL >= 1 && STEER_CHANNEL <= 16, "STEER_CHANNEL");
static_assert(THROTTLE_CHANNEL >= 1 && THROTTLE_CHANNEL <= 16, "THROTTLE_CHANNEL");
static_assert(ARM_CHANNEL >= 1 && ARM_CHANNEL <= 16, "ARM_CHANNEL");
static_assert(MODE_CHANNEL >= 1 && MODE_CHANNEL <= 16, "MODE_CHANNEL");
static_assert(
  ARM_CHANNEL != MODE_CHANNEL &&
  ARM_CHANNEL != STEER_CHANNEL && ARM_CHANNEL != THROTTLE_CHANNEL &&
  MODE_CHANNEL != STEER_CHANNEL && MODE_CHANNEL != THROTTLE_CHANNEL,
  "ARM, MODE, STEER and THROTTLE must be four different channels"
);
static_assert(PC_COMMAND_LIMIT >= 0 && PC_COMMAND_LIMIT <= 255, "PC_COMMAND_LIMIT");
static_assert(sizeof(WIFI_AP_PASSWORD) - 1 >= 8 && sizeof(WIFI_AP_PASSWORD) - 1 <= 63,
              "WIFI_AP_PASSWORD must be 8 to 63 characters");
static_assert(WIFI_AP_CHANNEL >= 1 && WIFI_AP_CHANNEL <= 13, "WIFI_AP_CHANNEL");


// =====================================================
// STATE
// =====================================================

uint16_t channels[16] = {0};

unsigned long lastValidRCFrame = 0;
bool rcFrameSeen = false;

unsigned long lastLinkStatsFrame = 0;
bool linkStatsSeen = false;
uint8_t uplinkLQ = 0;

uint8_t validFrameCount = 0;

// Debug counters
uint32_t goodFrames = 0;
uint32_t badCRC = 0;
uint32_t badLength = 0;

// ---- who has control ----

#define SOURCE_NONE   0
#define SOURCE_RADIO  1
#define SOURCE_PC     2

#define SWITCH_LOW      0
#define SWITCH_HIGH     1
#define SWITCH_INVALID  2

// Why the motors are not being driven right now.
#define BLOCK_NONE                 0   // a source has control
#define BLOCK_NO_RADIO             1   // no qualified radio link since boot
#define BLOCK_RADIO_LOST           2   // the radio link is down now
#define BLOCK_ARM_SWITCH_INVALID   3
#define BLOCK_DISARMED             4   // ARM switch off, ready to arm
#define BLOCK_ARM_ON_AT_BOOT       5   // ARM switch on, never seen off
#define BLOCK_REARM_REQUIRED       6   // ARM switch on after a fault
#define BLOCK_MODE_SWITCH_INVALID  7
#define BLOCK_WAIT_RADIO_NEUTRAL   8   // armed, RADIO: sticks not at zero yet
#define BLOCK_WAIT_PC_NEUTRAL      9   // armed, PC: no fresh zero command yet

// What caused a stop. Also used for the latched fault.
#define CAUSE_NONE                 0
#define CAUSE_DISARM               1
#define CAUSE_RADIO_LOST           2
#define CAUSE_PC_STALE             3
#define CAUSE_HANDOFF              4
#define CAUSE_ARM_SWITCH_INVALID   5
#define CAUSE_MODE_SWITCH_INVALID  6

bool radioLinkEverUp = false;

bool armed = false;         // only the ARM switch sets this
bool armOffSeen = false;    // ARM switch seen OFF since boot / last fault
uint8_t fault = CAUSE_NONE; // latched until the ARM switch is seen OFF

uint8_t target = SOURCE_NONE;   // source the MODE switch has selected
bool granted = false;           // ... and it has shown a fresh neutral
uint32_t handoffFrameMark = 0;  // goodFrames when the target last changed

uint8_t block = BLOCK_NO_RADIO;
uint8_t lastStop = CAUSE_NONE;  // sticky: cause of the most recent stop

// Changes on every arm, disarm, handoff and fault. A PC command must
// carry the current value, so nothing sent earlier can be accepted.
uint32_t token = 0;

// ---- PC commands ----

int pcThrottle = 0;
int pcSteer = 0;
unsigned long lastPcCommand = 0;   // last ACCEPTED command

unsigned long lastPcLine = 0;      // last well-formed command, accepted or not
bool pcLineSeen = false;
uint32_t pcBadLines = 0;

char pcLine[40];
uint8_t pcLineLength = 0;
bool pcLineDiscard = false;

// ---- which way the PC is talking ----
//
// The transport (and for Wi-Fi the address) that supplied the neutral
// command holds the PC input until the next handoff, disarm or fault.

#define PC_VIA_NONE  0
#define PC_VIA_USB   1
#define PC_VIA_WIFI  2

uint8_t pcVia = PC_VIA_NONE;
IPAddress pcPeerIp;
uint16_t pcPeerPort = 0;
uint32_t pcLastSequence = 0;        // Wi-Fi commands must count upwards
uint32_t pcOtherTransport = 0;      // commands ignored because another transport holds the input

WiFiUDP wifiUdp;
bool wifiApOk = false;
bool wifiUdpOk = false;
int wifiStations = 0;

IPAddress statusPeerIp;             // where status lines are sent over Wi-Fi
uint16_t statusPeerPort = 0;
bool statusPeerKnown = false;
unsigned long statusPeerAt = 0;

uint32_t wifiReceived = 0;
uint32_t wifiBad = 0;               // malformed, over-long or unprintable datagrams
uint32_t wifiOld = 0;               // repeated or out-of-order sequence numbers

unsigned long lastWifiPoll = 0;
unsigned long lastStationPoll = 0;
unsigned long lastNetLine = 0;
bool netSent = false;
uint32_t netSignature = 0;

// ---- outputs and diagnostics ----

int outLeft = 0;
int outRight = 0;

unsigned long lastStatus = 0;
unsigned long lastChannelLine = 0;
unsigned long lastHello = 0;
bool statusSent = false;
bool helloRequested = false;
uint32_t statusSignature = 0;

// =====================================================
// CRSF BUFFER
// =====================================================

uint8_t crsfBuffer[64];

uint8_t crsfIndex = 0;
uint8_t expectedFrameSize = 0;


// =====================================================
// MOTOR CONTROL
// =====================================================
//
// Same truth table as ELRS-Control.ino:
//   stop           = both inputs HIGH (DRV8833 brake)
//   left forward   = AIN1 PWM, AIN2 LOW     left reverse  = AIN1 LOW, AIN2 PWM
//   right forward  = BIN1 LOW, BIN2 PWM     right reverse = BIN1 PWM, BIN2 LOW
//
// STOP-PATH FIX. The original writes constant levels with digitalWrite
// and PWM with analogWrite on the same pin. On the ESP32 Arduino core
// 3.x, analogWrite hands the pin to the LEDC (PWM) peripheral, and
// digitalWrite on that pin is then ignored (esp32-hal-gpio.c,
// __digitalWrite: "IO is not set as GPIO"). The pin keeps its last PWM
// duty, so "brake" is not reached once a pin has been used for PWM.
//
// Here every level is written with analogWrite: 255 is a constant
// HIGH (ledcWrite turns 255 into full-on) and 0 a constant LOW.
// digitalWrite is never used on a motor pin.

void brakeMotors() {

  // DRV8833 active brake

  analogWrite(AIN1, 255);
  analogWrite(AIN2, 255);

  analogWrite(BIN1, 255);
  analogWrite(BIN2, 255);
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
    analogWrite(AIN1, 255);
    analogWrite(AIN2, 255);
  }

  else if (speed > 0) {

    // LEFT FORWARD
    analogWrite(AIN1, speed);
    analogWrite(AIN2, 0);
  }

  else {

    // LEFT REVERSE
    analogWrite(AIN1, 0);
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
    analogWrite(BIN1, 255);
    analogWrite(BIN2, 255);
  }

  else if (speed > 0) {

    // RIGHT FORWARD
    analogWrite(BIN1, 0);
    analogWrite(BIN2, speed);
  }

  else {

    // RIGHT REVERSE
    analogWrite(BIN1, -speed);
    analogWrite(BIN2, 0);
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
  rcFrameSeen = true;
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

    // A packed RC frame is exactly 24 long. Anything else is not
    // decoded and does not count as a valid frame.

    if (frameLength != CRSF_RC_FRAME_LENGTH) {

      badLength++;
      return;
    }

    decodeChannels(&crsfBuffer[3]);

    goodFrames++;

    if (validFrameCount < ARM_FRAME_COUNT) {
      validFrameCount++;
    }
  }

  else if (
    frameType == CRSF_FRAMETYPE_LINK_STATISTICS &&
    frameLength >= CRSF_LINK_STATS_MIN_LENGTH
  ) {

    // Payload: uplink RSSI 1, uplink RSSI 2, uplink LQ, ...

    uplinkLQ = crsfBuffer[5];

    lastLinkStatsFrame = millis();
    linkStatsSeen = true;
  }
}


// =====================================================
// ROBUST CRSF PARSER (unchanged from ELRS-Control.ino)
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
// CHANNEL NORMALISATION (unchanged from ELRS-Control.ino)
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
// DIFFERENTIAL DRIVE (unchanged from ELRS-Control.ino)
// =====================================================

void driveMixed(int throttle, int steering) {

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

  setLeftMotor(leftSpeed);
  setRightMotor(rightSpeed);

  outLeft = leftSpeed;
  outRight = rightSpeed;
}


// =====================================================
// SWITCH CHANNELS
// =====================================================

int switchPosition(int value) {

  if (value < SWITCH_VALID_MIN || value > SWITCH_VALID_MAX) {
    return SWITCH_INVALID;
  }

  if (value <= SWITCH_LOW_MAX) {
    return SWITCH_LOW;
  }

  if (value >= SWITCH_HIGH_MIN) {
    return SWITCH_HIGH;
  }

  return SWITCH_INVALID;
}


// =====================================================
// TAKING CONTROL AWAY
// =====================================================
//
// Every path that removes control goes through one of these two.
// Both brake first, forget the old command and change the token.

void clearCommands() {

  brakeMotors();

  outLeft = 0;
  outRight = 0;

  pcThrottle = 0;
  pcSteer = 0;
  pcVia = PC_VIA_NONE;

  granted = false;

  token++;
}


// Disarm. Motion needs the ARM switch again afterwards.

void disarm(uint8_t cause) {

  if (armed) {
    lastStop = cause;
  }

  if (armed || granted || target != SOURCE_NONE) {
    clearCommands();
  }

  armed = false;
  target = SOURCE_NONE;
}


// Hand over to another source (or to none). The new source has to
// show a neutral input that arrives after this point.

void changeTarget(uint8_t newTarget, uint8_t cause) {

  if (granted) {
    lastStop = cause;
  }

  clearCommands();

  target = newTarget;
  handoffFrameMark = goodFrames;
}


// =====================================================
// PC COMMAND LINES
// =====================================================
//
//   D,<token>,<throttle>,<steer>\n
//
// token:    the "tok=" value from the latest ST line
// throttle: -255..255, positive = forward
// steer:    -255..255, positive = right
//
// A command is accepted only while armed, with PC selected, and with
// the current token. The first accepted command after a handoff must
// be 0,0. Nothing else on this port does anything, except "?" which
// asks for the HELLO line.

bool parseUnsigned(const char *&p, uint32_t &value) {

  if (*p < '0' || *p > '9') {
    return false;
  }

  uint32_t result = 0;

  while (*p >= '0' && *p <= '9') {

    uint32_t digit = (uint32_t)(*p - '0');

    if (result > (0xFFFFFFFFUL - digit) / 10) {
      return false;
    }

    result = result * 10 + digit;
    p++;
  }

  value = result;
  return true;
}


bool parseCommandValue(const char *&p, int &value) {

  bool negative = false;

  if (*p == '-') {
    negative = true;
    p++;
  }

  uint32_t magnitude;

  if (!parseUnsigned(p, magnitude) || magnitude > 255) {
    return false;
  }

  value = negative ? -(int)magnitude : (int)magnitude;
  return true;
}


bool parsePcCommandText(
  const char *text, bool withSequence,
  uint32_t &lineToken, int &throttle, int &steer, uint32_t &sequence
) {

  const char *p = text;

  if (p[0] != 'D' || p[1] != ',') {
    return false;
  }

  p += 2;

  if (!parseUnsigned(p, lineToken) || *p++ != ',') {
    return false;
  }

  if (!parseCommandValue(p, throttle) || *p++ != ',') {
    return false;
  }

  if (!parseCommandValue(p, steer)) {
    return false;
  }

  if (withSequence) {

    if (*p++ != ',' || !parseUnsigned(p, sequence)) {
      return false;
    }
  }

  return *p == '\0';
}


// The one place that decides whether a PC command counts, whichever
// way it arrived. The rules are those of the USB version; the lines
// marked WIFI are the only additions.

void acceptPcCommand(
  unsigned long now, uint8_t via,
  uint32_t lineToken, int throttle, int steer,
  uint32_t sequence, IPAddress peerIp, uint16_t peerPort
) {

  lastPcLine = now;
  pcLineSeen = true;

  if (!armed || target != SOURCE_PC || lineToken != token) {
    return;
  }

  throttle = constrain(throttle, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT);
  steer = constrain(steer, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT);

  if (!granted) {

    // Waiting for neutral: only 0,0 is accepted, and it grants control.

    if (throttle != 0 || steer != 0) {
      return;
    }

    granted = true;

    // WIFI: whoever supplied the neutral holds the PC input.
    pcVia = via;
    pcPeerIp = peerIp;
    pcPeerPort = peerPort;
    pcLastSequence = sequence;
  }

  else {

    // WIFI: USB and Wi-Fi never command together. The other one is
    // ignored, and does not keep the link alive either.

    if (via != pcVia) {

      pcOtherTransport++;
      return;
    }

    if (via == PC_VIA_WIFI) {

      if (peerIp != pcPeerIp || peerPort != pcPeerPort) {

        pcOtherTransport++;
        return;
      }

      // A repeated or late datagram is never applied.

      if (sequence <= pcLastSequence) {

        wifiOld++;
        return;
      }

      pcLastSequence = sequence;
    }
  }

  pcThrottle = throttle;
  pcSteer = steer;
  lastPcCommand = now;
}


// USB:  D,<token>,<throttle>,<steer>

void handlePcLine(unsigned long now) {

  if (pcLine[0] == '?' && pcLine[1] == '\0') {

    helloRequested = true;
    return;
  }

  uint32_t lineToken;
  int throttle;
  int steer;
  uint32_t sequence = 0;

  if (!parsePcCommandText(pcLine, false, lineToken, throttle, steer, sequence)) {

    pcBadLines++;
    return;
  }

  acceptPcCommand(now, PC_VIA_USB, lineToken, throttle, steer, 0, IPAddress(), 0);
}


// Status lines go to whoever last sent a well-formed datagram. While a
// Wi-Fi peer holds the PC input, they keep going to that peer.

void rememberStatusPeer(IPAddress peerIp, uint16_t peerPort, unsigned long now) {

  if (
    granted && pcVia == PC_VIA_WIFI &&
    (peerIp != pcPeerIp || peerPort != pcPeerPort)
  ) {
    return;
  }

  statusPeerIp = peerIp;
  statusPeerPort = peerPort;
  statusPeerKnown = true;
  statusPeerAt = now;
}


// Wi-Fi: one datagram = one line.
//
//   D,<token>,<throttle>,<steer>,<sequence>      sequence counts up by one per command
//   ?                                            ask for the HELLO line
//
// Never waits: parsePacket() returns at once when nothing has arrived.

void readWifi(unsigned long now) {

  if (!wifiUdpOk || now == lastWifiPoll) {
    return;
  }

  lastWifiPoll = now;

  for (int i = 0; i < WIFI_MAX_DATAGRAMS; i++) {

    int size = wifiUdp.parsePacket();

    if (size <= 0) {
      break;
    }

    char text[48];

    int length = wifiUdp.read((uint8_t *)text, sizeof(text) - 1);

    IPAddress peerIp = wifiUdp.remoteIP();
    uint16_t peerPort = wifiUdp.remotePort();

    // Drop whatever was not read, or the next datagram is never seen.
    wifiUdp.clear();

    wifiReceived++;

    if (length <= 0 || size > (int)sizeof(text) - 1) {

      wifiBad++;
      continue;
    }

    if (text[length - 1] == '\n') {
      length--;
    }

    text[length] = '\0';

    bool printable = length > 0;

    for (int k = 0; k < length; k++) {

      if (text[k] < 32 || text[k] > 126) {
        printable = false;
      }
    }

    if (!printable) {

      wifiBad++;
      continue;
    }

    if (text[0] == '?' && text[1] == '\0') {

      rememberStatusPeer(peerIp, peerPort, now);

      helloRequested = true;
      continue;
    }

    uint32_t lineToken;
    int throttle;
    int steer;
    uint32_t sequence;

    if (!parsePcCommandText(text, true, lineToken, throttle, steer, sequence)) {

      wifiBad++;
      continue;
    }

    rememberStatusPeer(peerIp, peerPort, now);

    acceptPcCommand(now, PC_VIA_WIFI, lineToken, throttle, steer, sequence, peerIp, peerPort);
  }
}


void readPc(unsigned long now) {

  // Bounded, so a flood on USB cannot starve the radio.

  for (int i = 0; i < 256 && USB_PORT.available() > 0; i++) {

    int b = USB_PORT.read();

    if (b < 0) {
      break;
    }

    if (b == '\n') {

      if (!pcLineDiscard) {

        pcLine[pcLineLength] = '\0';
        handlePcLine(now);
      }

      else {
        pcBadLines++;
      }

      pcLineLength = 0;
      pcLineDiscard = false;
    }

    else if (!pcLineDiscard) {

      // Over-long lines and anything unprintable (including '\r')
      // spoil the whole line.

      if (pcLineLength >= sizeof(pcLine) - 1 || b < 32 || b > 126) {
        pcLineDiscard = true;
      }

      else {
        pcLine[pcLineLength++] = (char)b;
      }
    }
  }
}


// =====================================================
// DIAGNOSTIC LINES
// =====================================================

const char *sourceName(uint8_t source) {

  switch (source) {
    case SOURCE_RADIO: return "RADIO";
    case SOURCE_PC:    return "PC";
    default:           return "NONE";
  }
}


const char *blockName(uint8_t value) {

  switch (value) {
    case BLOCK_NONE:                return "NONE";
    case BLOCK_NO_RADIO:            return "NO_RADIO";
    case BLOCK_RADIO_LOST:          return "RADIO_LOST";
    case BLOCK_ARM_SWITCH_INVALID:  return "ARM_SWITCH_INVALID";
    case BLOCK_DISARMED:            return "DISARMED";
    case BLOCK_ARM_ON_AT_BOOT:      return "ARM_ON_AT_BOOT";
    case BLOCK_REARM_REQUIRED:      return "REARM_REQUIRED";
    case BLOCK_MODE_SWITCH_INVALID: return "MODE_SWITCH_INVALID";
    case BLOCK_WAIT_RADIO_NEUTRAL:  return "WAIT_RADIO_NEUTRAL";
    case BLOCK_WAIT_PC_NEUTRAL:     return "WAIT_PC_NEUTRAL";
    default:                        return "UNKNOWN";
  }
}


const char *causeName(uint8_t value) {

  switch (value) {
    case CAUSE_NONE:                return "NONE";
    case CAUSE_DISARM:              return "DISARM";
    case CAUSE_RADIO_LOST:          return "RADIO_LOST";
    case CAUSE_PC_STALE:            return "PC_STALE";
    case CAUSE_HANDOFF:             return "HANDOFF";
    case CAUSE_ARM_SWITCH_INVALID:  return "ARM_SWITCH_INVALID";
    case CAUSE_MODE_SWITCH_INVALID: return "MODE_SWITCH_INVALID";
    default:                        return "UNKNOWN";
  }
}


const char *armSwitchName(int position) {

  if (position == SWITCH_INVALID) {
    return "INVALID";
  }

  return position == (ARM_WHEN_HIGH ? SWITCH_HIGH : SWITCH_LOW) ? "ARM" : "DISARM";
}


const char *modeSwitchName(int position) {

  if (position == SWITCH_INVALID) {
    return "INVALID";
  }

  return position == (PC_WHEN_HIGH ? SWITCH_HIGH : SWITCH_LOW) ? "PC" : "RADIO";
}


// Whole line or nothing, and never wait for the PC.
// The same line goes to USB (if it has room) and to the Wi-Fi peer
// (if there is one). Neither needs the other.

void sendLine(const char *line, int size) {

  if (size <= 0 || size >= 250) {
    return;
  }

  if (USB_PORT.availableForWrite() >= size) {
    USB_PORT.write((const uint8_t *)line, (size_t)size);
  }

  if (wifiUdpOk && statusPeerKnown) {

    // The socket is non-blocking; a datagram that cannot be sent is dropped.

    if (wifiUdp.beginPacket(statusPeerIp, statusPeerPort)) {

      wifiUdp.write((const uint8_t *)line, (size_t)size);
      wifiUdp.endPacket();
    }
  }
}


long ageOf(bool seen, unsigned long now, unsigned long then) {

  if (!seen) {
    return -1;
  }

  return (long)min(now - then, 99999UL);
}


void sendHello() {

  char out[240];

  int size = snprintf(
    out, sizeof(out),
    "HELLO fw=autobot-handoff-wifi-1 arm=CH%d:%s pc=CH%d:%s "
    "pc_limit=%d pc_timeout_ms=%d rc_timeout_ms=%d "
    "wifi_ap=%s wifi_ok=%d wifi_channel=%d udp_port=%d\n",
    ARM_CHANNEL, ARM_WHEN_HIGH ? "HIGH" : "LOW",
    MODE_CHANNEL, PC_WHEN_HIGH ? "HIGH" : "LOW",
    PC_COMMAND_LIMIT, PC_TIMEOUT, FAILSAFE_TIMEOUT,
    WIFI_AP_SSID, (wifiApOk && wifiUdpOk) ? 1 : 0, WIFI_AP_CHANNEL, WIFI_UDP_PORT
  );

  sendLine(out, size);
}


// NET: the PC link. Sent on every change and twice a second.
//
//   ap      1 = the robot's access point and UDP port are up
//   sta     laptops joined to the access point (0 or 1)
//   peer    address status lines are sent to over Wi-Fi, or NONE
//   pc_via  which transport holds the PC input now: NONE, USB or WIFI
//   rx      datagrams received      bad    malformed datagrams
//   old     repeated / late datagrams ignored
//   other   commands ignored because the other transport holds the input

void sendNet(unsigned long now) {

  char peer[32];

  if (statusPeerKnown) {

    snprintf(
      peer, sizeof(peer), "%u.%u.%u.%u:%u",
      (unsigned)statusPeerIp[0], (unsigned)statusPeerIp[1],
      (unsigned)statusPeerIp[2], (unsigned)statusPeerIp[3],
      (unsigned)statusPeerPort
    );
  }

  else {
    snprintf(peer, sizeof(peer), "NONE");
  }

  char out[200];

  int size = snprintf(
    out, sizeof(out),
    "NET ap=%d sta=%d peer=%s pc_via=%s rx=%lu bad=%lu old=%lu other=%lu\n",
    (wifiApOk && wifiUdpOk) ? 1 : 0, wifiStations, peer,
    pcVia == PC_VIA_WIFI ? "WIFI" : pcVia == PC_VIA_USB ? "USB" : "NONE",
    (unsigned long)wifiReceived, (unsigned long)wifiBad,
    (unsigned long)wifiOld, (unsigned long)pcOtherTransport
  );

  sendLine(out, size);
}


// ST: who has control and why. Sent on every change and as a heartbeat.
//
//   link   1 = qualified radio link
//   armed  1 = armed by the ARM switch
//   req    what the MODE switch asks for (NONE = no link or invalid)
//   act    the source whose commands reach the motors now
//   pc_ok  1 = the firmware is accepting PC drive commands
//   block  why nothing is driving (NONE when a source has control)
//   fault  latched; needs ARM off then on
//   stop   cause of the most recent stop
//   L, R   motor outputs, -255..255
//   pc_age ms since the last well-formed PC command line, -1 = never

void sendStatus(unsigned long now, bool link, uint8_t requested) {

  char out[220];

  bool driving = armed && granted;

  int size = snprintf(
    out, sizeof(out),
    "ST t=%lu tok=%lu link=%d armed=%d req=%s act=%s pc_ok=%d "
    "block=%s fault=%s stop=%s L=%d R=%d pc_age=%ld\n",
    now, (unsigned long)token,
    link ? 1 : 0, armed ? 1 : 0,
    sourceName(requested),
    sourceName(driving ? target : SOURCE_NONE),
    (driving && target == SOURCE_PC) ? 1 : 0,
    blockName(block), causeName(fault), causeName(lastStop),
    outLeft, outRight,
    ageOf(pcLineSeen, now, lastPcLine)
  );

  sendLine(out, size);
}


// CH: raw radio values, for finding your switches. The ARM and MODE
// columns show the raw value and how it decodes.

void sendChannels(unsigned long now) {

  char out[240];

  int armRaw = channels[ARM_CHANNEL - 1];
  int modeRaw = channels[MODE_CHANNEL - 1];

  int size = snprintf(
    out, sizeof(out),
    "CH rc_age=%ld frames=%lu crc=%lu len=%lu lq=%d "
    "ch1=%u ch2=%u ch5=%u ch6=%u ch7=%u ch8=%u ch9=%u ch10=%u "
    "arm=CH%d:%d:%s mode=CH%d:%d:%s steer=%d thr=%d pc_bad=%lu\n",
    ageOf(rcFrameSeen, now, lastValidRCFrame),
    (unsigned long)goodFrames, (unsigned long)badCRC, (unsigned long)badLength,
    linkStatsSeen ? (int)uplinkLQ : -1,
    (unsigned)channels[0], (unsigned)channels[1],
    (unsigned)channels[4], (unsigned)channels[5],
    (unsigned)channels[6], (unsigned)channels[7], (unsigned)channels[8],
    (unsigned)channels[9],
    ARM_CHANNEL, armRaw, armSwitchName(switchPosition(armRaw)),
    MODE_CHANNEL, modeRaw, modeSwitchName(switchPosition(modeRaw)),
    normalizeChannel(channels[STEER_CHANNEL - 1]),
    normalizeChannel(channels[THROTTLE_CHANNEL - 1]),
    (unsigned long)pcBadLines
  );

  sendLine(out, size);
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  // -----------------------------
  // MOTORS FIRST
  // -----------------------------
  //
  // Attach all four pins to PWM at 0 (LOW, the level they already
  // have after reset), then set brake. Going from all-LOW (coast) to
  // all-HIGH (brake) the two inputs of a motor may differ for up to
  // one PWM period (1 ms).

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);

  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  analogWrite(AIN1, 0);
  analogWrite(AIN2, 0);

  analogWrite(BIN1, 0);
  analogWrite(BIN2, 0);

  brakeMotors();


  // -----------------------------
  // USB SERIAL
  // -----------------------------

  USB_PORT.begin(115200);

  delay(500);


  // -----------------------------
  // ELRS UART
  // -----------------------------

  // Room for about 25 ms of radio bytes, in case the Wi-Fi task holds
  // the processor for a moment. Must be set before begin().

  ELRS.setRxBufferSize(1024);

  ELRS.begin(
    CRSF_BAUD,
    SERIAL_8N1,
    ELRS_RX_PIN,
    -1
  );


  // -----------------------------
  // WI-FI ACCESS POINT
  // -----------------------------
  //
  // The motors are already braked. If this fails the sketch carries
  // on: the radio and USB work exactly as in the USB version.

  WiFi.persistent(false);
  WiFi.mode(WIFI_AP);

  wifiApOk = WiFi.softAP(
    WIFI_AP_SSID,
    WIFI_AP_PASSWORD,
    WIFI_AP_CHANNEL,
    0,      // name is visible
    1       // one laptop at a time
  );

  WiFi.setSleep(false);

#if WIFI_REDUCED_TX_POWER
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
#endif

  wifiUdpOk = wifiApOk && wifiUdp.begin(WIFI_UDP_PORT);


  // A different token every boot, so a PC command from before a
  // reset can never match.

  token = esp_random();

  sendHello();
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop() {

  readCRSF();

  unsigned long now = millis();

  // PC lines are judged against the state as it stands. Anything the
  // radio decides below overrides what they did.

  readPc(now);
  readWifi(now);


  // ===================================================
  // RADIO LINK
  // ===================================================
  //
  // Qualified = fresh CRC-valid RC frames, ARM_FRAME_COUNT of them in
  // a row since the last gap, and no fresh report of 0 % link quality.

  bool rcFresh =
    rcFrameSeen &&
    now - lastValidRCFrame <= FAILSAFE_TIMEOUT;

  if (!rcFresh) {
    validFrameCount = 0;
  }

  bool lqZero =
    LINK_LOSS_ON_LQ_ZERO &&
    linkStatsSeen &&
    now - lastLinkStatsFrame <= LINK_STATS_MAX_AGE &&
    uplinkLQ == 0;

  bool link =
    rcFresh &&
    validFrameCount >= ARM_FRAME_COUNT &&
    !lqZero;


  // ===================================================
  // RADIO INPUT
  // ===================================================

  int steering =
    normalizeChannel(channels[STEER_CHANNEL - 1]);

  int throttle =
    normalizeChannel(channels[THROTTLE_CHANNEL - 1]);

  int armSwitch =
    link ? switchPosition(channels[ARM_CHANNEL - 1]) : SWITCH_INVALID;

  int modeSwitch =
    link ? switchPosition(channels[MODE_CHANNEL - 1]) : SWITCH_INVALID;

  uint8_t requested = SOURCE_NONE;

  if (modeSwitch != SWITCH_INVALID) {

    requested =
      modeSwitch == (PC_WHEN_HIGH ? SWITCH_HIGH : SWITCH_LOW)
        ? SOURCE_PC
        : SOURCE_RADIO;
  }


  // ===================================================
  // ARM AUTHORITY (RadioMaster only)
  // ===================================================

  block = BLOCK_NONE;

  if (!link) {

    // Radio lost, in either mode. Needs ARM off then on afterwards.

    if (radioLinkEverUp) {
      fault = CAUSE_RADIO_LOST;
    }

    disarm(CAUSE_RADIO_LOST);
    armOffSeen = false;

    block = radioLinkEverUp ? BLOCK_RADIO_LOST : BLOCK_NO_RADIO;
  }

  else if (armSwitch == SWITCH_INVALID) {

    radioLinkEverUp = true;

    fault = CAUSE_ARM_SWITCH_INVALID;

    disarm(CAUSE_ARM_SWITCH_INVALID);
    armOffSeen = false;

    block = BLOCK_ARM_SWITCH_INVALID;
  }

  else if (armSwitch != (ARM_WHEN_HIGH ? SWITCH_HIGH : SWITCH_LOW)) {

    // ARM switch off. This is the only thing that clears a fault.

    radioLinkEverUp = true;

    disarm(CAUSE_DISARM);
    armOffSeen = true;
    fault = CAUSE_NONE;

    block = BLOCK_DISARMED;
  }

  else {

    // ARM switch on.

    radioLinkEverUp = true;

    if (!armed) {

      if (armOffSeen) {

        // Off, then on, on a live link: the explicit arm action.

        armed = true;
        target = SOURCE_NONE;
        granted = false;
        token++;
      }

      else {

        block =
          fault != CAUSE_NONE
            ? BLOCK_REARM_REQUIRED
            : BLOCK_ARM_ON_AT_BOOT;
      }
    }
  }


  // ===================================================
  // HANDOFF: NONE / RADIO / PC
  // ===================================================

  if (armed) {

    if (requested == SOURCE_NONE) {

      // MODE switch invalid: nobody drives.

      if (target != SOURCE_NONE) {
        changeTarget(SOURCE_NONE, CAUSE_MODE_SWITCH_INVALID);
      }

      block = BLOCK_MODE_SWITCH_INVALID;
    }

    else {

      if (requested != target) {

        // Zero first, forget the old command, new token.
        changeTarget(requested, CAUSE_HANDOFF);
      }

      if (target == SOURCE_PC) {

        if (granted && now - lastPcCommand >= PC_TIMEOUT) {

          // PC commands stopped (stale, client closed, USB pulled).
          // No fallback to the radio sticks: stop and latch.

          fault = CAUSE_PC_STALE;

          disarm(CAUSE_PC_STALE);
          armOffSeen = false;

          block = BLOCK_REARM_REQUIRED;
        }

        else if (!granted) {

          // readPc() grants on a 0,0 command with the current token.
          block = BLOCK_WAIT_PC_NEUTRAL;
        }
      }

      else if (!granted) {

        // RADIO: both normalised sticks exactly zero, in a frame
        // that arrived after the handoff.

        if (
          goodFrames != handoffFrameMark &&
          throttle == 0 &&
          steering == 0
        ) {
          granted = true;
        }

        else {
          block = BLOCK_WAIT_RADIO_NEUTRAL;
        }
      }
    }
  }


  // ===================================================
  // DRIVE
  // ===================================================

  if (!armed || !granted || block != BLOCK_NONE) {

    brakeMotors();

    outLeft = 0;
    outRight = 0;
  }

  else if (target == SOURCE_PC) {

    driveMixed(pcThrottle, pcSteer);
  }

  else if (target == SOURCE_RADIO) {

    driveMixed(throttle, steering);
  }

  else {

    brakeMotors();

    outLeft = 0;
    outRight = 0;
  }


  // ===================================================
  // DIAGNOSTICS (after the motors are set)
  // ===================================================

  uint32_t signature =
    token * 2654435761UL +
    ((uint32_t)link |
     ((uint32_t)armed << 1) |
     ((uint32_t)granted << 2) |
     ((uint32_t)requested << 3) |
     ((uint32_t)target << 5) |
     ((uint32_t)block << 7) |
     ((uint32_t)fault << 11) |
     ((uint32_t)lastStop << 14));

  bool changed = !statusSent || signature != statusSignature;

  if (
    (changed && (!statusSent || now - lastStatus >= STATUS_MIN_GAP_MS)) ||
    now - lastStatus >= STATUS_PERIOD_MS
  ) {

    lastStatus = now;
    statusSent = true;
    statusSignature = signature;

    sendStatus(now, link, requested);
  }

  // Kept 5 ms clear of an ST line, so both fit in the USB buffer.

  if (
    now - lastStatus >= 5 &&
    now - lastNetLine >= 5 &&
    now - lastChannelLine >=
    (unsigned long)(armed ? CHANNEL_PERIOD_ARMED_MS : CHANNEL_PERIOD_IDLE_MS)
  ) {

    lastChannelLine = now;

    sendChannels(now);
  }

  if (helloRequested && now - lastHello >= 500) {

    helloRequested = false;
    lastHello = now;

    sendHello();
  }


  // ===================================================
  // WI-FI LINK STATE
  // ===================================================

  if (statusPeerKnown && now - statusPeerAt > WIFI_PEER_TIMEOUT) {
    statusPeerKnown = false;
  }

  if (wifiApOk && now - lastStationPoll >= 250) {

    lastStationPoll = now;
    wifiStations = WiFi.softAPgetStationNum();
  }

  uint32_t net =
    (uint32_t)wifiStations |
    ((uint32_t)pcVia << 4) |
    ((uint32_t)statusPeerKnown << 6) |
    ((uint32_t)statusPeerPort << 8);

  bool netChanged = !netSent || net != netSignature;

  if (
    now - lastStatus >= 5 &&
    now - lastChannelLine >= 5 &&
    ((netChanged && (!netSent || now - lastNetLine >= STATUS_MIN_GAP_MS)) ||
     now - lastNetLine >= 500)
  ) {

    lastNetLine = now;
    netSent = true;
    netSignature = net;

    sendNet(now);
  }
}
