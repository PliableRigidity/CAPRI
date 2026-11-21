// Host-side test of ELRS-Handoff-WiFi.ino: every scenario of the USB version, plus the Wi-Fi link.
// (Copied from ELRS-Handoff-Bench/host_test/test_handoff.cpp and extended.)
// Host-side test of the sketch. It compiles the real sketch (and
// the untouched original, ELRS-Control.ino, in its own namespace) against
// mock/Arduino.h, feeds CRSF bytes to the receiver UART and command lines to
// USB, runs one loop() per simulated millisecond and reads the motor pins back.
//
// It proves the logic on a model of the board. It does not prove real pin
// levels, receiver behaviour or USB timing: those need the bench.
//
// One scenario per process, so every scenario starts from a fresh boot:
//   test_handoff.exe --list
//   test_handoff.exe <scenario>
//   test_handoff.exe cosim        (driven over stdin by tests/test_handoff_wifi.py)
//
// Build and run with:  python -m pytest tests/test_handoff_wifi.py

#include "Arduino.h"

// Both found through /I, so a mutated copy of the sketch can be swapped in.
#include "ELRS-Handoff-WiFi.ino"

// Shared #defines (pins, baud, calibration, timeouts) must be token-identical
// in both sketches: the build uses /we4005, so a different value is an error.
namespace original {
#include "ELRS-Control.ino"
}

#include <cstdarg>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>

// ---------------------------------------------------------------- reporting

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char* format, ...) {
  char text[320];
  va_list args;
  va_start(args, format);
  std::vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  ++checks;
  if (!ok) ++failures;
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", text);
}

// ---------------------------------------------------------------- CRSF out

static uint8_t crcOf(const std::vector<uint8_t>& bytes) {
  uint8_t crc = 0;
  for (uint8_t b : bytes) {
    crc ^= b;
    for (int i = 0; i < 8; ++i) crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0xD5) : static_cast<uint8_t>(crc << 1);
  }
  return crc;
}

static void pushFrame(std::deque<uint8_t>& uart, uint8_t type, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> body{type};
  body.insert(body.end(), payload.begin(), payload.end());
  uart.push_back(0xC8);
  uart.push_back(static_cast<uint8_t>(body.size() + 1));
  for (uint8_t b : body) uart.push_back(b);
  uart.push_back(crcOf(body));
}

static std::vector<uint8_t> packChannels(const int ch[16]) {
  std::vector<uint8_t> payload(22, 0);
  int bit = 0;
  for (int c = 0; c < 16; ++c) {
    for (int i = 0; i < 11; ++i, ++bit) {
      if (ch[c] & (1 << i)) payload[bit / 8] |= static_cast<uint8_t>(1 << (bit % 8));
    }
  }
  return payload;
}

// ---------------------------------------------------------------- expected drive

// Written from the description of the original (calibration, deadzone, mix),
// independently of the sketch code. Scenario drive_matches_original also
// compares against the original sketch itself.
static int expectNorm(int value) {
  if (std::abs(value - 992) <= 45) return 0;
  long out = value > 992 ? (static_cast<long>(value) - 1037) * 255 / (1811 - 1037)
                         : (static_cast<long>(value) - 174) * 255 / (947 - 174) - 255;
  return static_cast<int>(std::max(-255L, std::min(255L, out)));
}

static void expectMix(int throttle, int steer, int& left, int& right) {
  left = throttle + steer;
  right = throttle - steer;
  const int top = std::max(std::abs(left), std::abs(right));
  if (top > 255) {
    left = left * 255 / top;
    right = right * 255 / top;
  }
}

// ---------------------------------------------------------------- motors

struct Motors {
  bool valid = false;  // pins form a legal DRV8833 state on both sides
  bool leftBrake = false, rightBrake = false;
  int left = 0, right = 0;  // -255..255, positive = forward
  bool stopped() const { return valid && leftBrake && rightBrake; }
};

static bool decodeSide(int forwardPin, int reversePin, bool& brake, int& speed) {
  const int f = mock::pinOut(forwardPin);
  const int r = mock::pinOut(reversePin);
  brake = false;
  speed = 0;
  if (f == 256 && r == 256) { brake = true; return true; }
  if (r == 0 && f > 0) { speed = f == 256 ? 255 : f; return true; }
  if (f == 0 && r > 0) { speed = -(r == 256 ? 255 : r); return true; }
  return false;  // coast, floating, or PWM on both inputs
}

static Motors motors() {
  Motors m;
  const bool l = decodeSide(AIN1, AIN2, m.leftBrake, m.left);    // left forward = AIN1 PWM
  const bool r = decodeSide(BIN2, BIN1, m.rightBrake, m.right);  // right forward = BIN2 PWM
  m.valid = l && r;
  return m;
}

static std::string pinText(int pin) {
  const int out = mock::pinOut(pin);
  if (out < 0) return "floating";
  if (out == 0) return "LOW";
  if (out == 256) return "HIGH";
  return "PWM " + std::to_string(out) + "/256";
}

// ---------------------------------------------------------------- the rig

static const int SW_LOW_RAW = 191;    // two-position switch, low
static const int SW_HIGH_RAW = 1792;  // two-position switch, high
static const int ARM_ON = ARM_WHEN_HIGH ? SW_HIGH_RAW : SW_LOW_RAW;
static const int ARM_OFF = ARM_WHEN_HIGH ? SW_LOW_RAW : SW_HIGH_RAW;
static const int MODE_PC = PC_WHEN_HIGH ? SW_HIGH_RAW : SW_LOW_RAW;
static const int MODE_RADIO = PC_WHEN_HIGH ? SW_LOW_RAW : SW_HIGH_RAW;

enum Band { BAND_LOW, BAND_HIGH, BAND_BAD };
static Band band(int raw) {
  if (raw < 100 || raw > 1950) return BAND_BAD;
  if (raw <= 600) return BAND_LOW;
  if (raw >= 1400) return BAND_HIGH;
  return BAND_BAD;
}

enum PcKind { PC_SILENT, PC_FOLLOW, PC_FIXED_TOKEN, PC_SCRIPT };

struct Rig {
  // Transmitter and receiver
  bool radioOn = false;
  bool statsOn = false;
  bool junk = false;
  uint8_t lq = 100;
  int steer = 992, throttle = 992;
  int armRaw = ARM_OFF, modeRaw = MODE_RADIO;

  // PC side
  PcKind pc = PC_SILENT;
  int pcThrottle = 0, pcSteer = 0;
  unsigned long fixedToken = 0;
  std::function<std::string(Rig&)> script;

  // Seen on USB
  std::map<std::string, std::string> st, ch;
  std::string hello;
  std::string line;
  std::string capture;
  int stLines = 0, chLines = 0, helloLines = 0, badLines = 0, longest = 0;
  std::string seenAct;  // every act= value seen since it was last cleared

  // Oracles: things that must be true whenever a wheel is driven
  int sent[16] = {0};
  bool everSent = false;
  unsigned long lastFrameMs = 0;
  int framesSinceGap = 0;
  bool armOffSinceLink = false;
  bool neutralSinceChange = false;
  bool zeroPcSinceChange = false;
  bool pcSentEver = false;
  unsigned long lastPcSentMs = 0;
  int lastPcT = 0, lastPcS = 0, prevPcT = 0, prevPcS = 0;
  int violations = 0;
  uint32_t lcg = 12345;

  unsigned long token() const {
    auto it = st.find("tok");
    return it == st.end() ? 0UL : std::strtoul(it->second.c_str(), nullptr, 10);
  }
  std::string s(const char* key) const {
    auto it = st.find(key);
    return it == st.end() ? "?" : it->second;
  }
  bool is(const char* key, const char* value) const { return s(key) == value; }

  void sendPc(const std::string& text) {
    for (char c : text) HWCDCSerial.rx.push_back(static_cast<uint8_t>(c));
  }

  void sendDrive(unsigned long tok, int t, int steerValue) {
    char text[64];
    std::snprintf(text, sizeof(text), "D,%lu,%d,%d\n", tok, t, steerValue);
    sendPc(text);
    prevPcT = lastPcT; prevPcS = lastPcS;
    lastPcT = t; lastPcS = steerValue;
    lastPcSentMs = mock::now_ms;
    pcSentEver = true;
    if (t == 0 && steerValue == 0) zeroPcSinceChange = true;
  }

  // ---- the Wi-Fi laptop ----
  enum WifiKind { WIFI_OFF, WIFI_FOLLOW, WIFI_FIXED_TOKEN };
  WifiKind wifi = WIFI_OFF;
  int wifiThrottle = 0, wifiSteer = 0;
  unsigned long wifiFixedToken = 0;
  uint32_t wifiSeq = 0;
  bool wifiReachable = true;                      // false = the laptop is not on the robot's network
  int stationsForced = -1;
  IPAddress wifiIp{192, 168, 4, 2};
  uint16_t wifiPort = 50123;
  std::map<std::string, std::string> wst, wnet, net;   // last ST / NET over Wi-Fi; last NET over USB
  std::string wifiHello, wifiCapture, wifiLine;
  int wifiStLines = 0, wifiNetLines = 0, wifiChLines = 0, wifiHelloLines = 0, wifiBadLines = 0;
  int wifiStray = 0, wifiOutCount = 0, netLines = 0;
  std::vector<std::string> wifiSent;              // every command datagram the laptop produced, verbatim
  int lastWifiT = 0, lastWifiS = 0, prevWifiT = 0, prevWifiS = 0;

  unsigned long wifiToken() const {
    auto it = wst.find("tok");
    return it == wst.end() ? 0UL : std::strtoul(it->second.c_str(), nullptr, 10);
  }
  std::string ws(const char* key) const {
    auto it = wst.find(key);
    return it == wst.end() ? "?" : it->second;
  }
  bool wis(const char* key, const char* value) const { return ws(key) == value; }
  std::string wn(const char* key) const {
    auto it = wnet.find(key);
    return it == wnet.end() ? "?" : it->second;
  }
  std::string un(const char* key) const {
    auto it = net.find(key);
    return it == net.end() ? "?" : it->second;
  }

  void sendWifiRaw(const std::string& text, IPAddress ip, uint16_t port) {
    if (wifiReachable) mock::wifi_in.push_back(mock::Datagram{ip, port, text});
  }
  void sendWifiRaw(const std::string& text) { sendWifiRaw(text, wifiIp, wifiPort); }

  void sendWifiDrive(unsigned long tok, int t, int steerValue) {
    char text[96];
    std::snprintf(text, sizeof(text), "D,%lu,%d,%d,%lu\n", tok, t, steerValue, static_cast<unsigned long>(++wifiSeq));
    wifiSent.push_back(text);
    if (!wifiReachable) return;                   // sent into the air; never arrives
    sendWifiRaw(text);
    prevWifiT = lastWifiT; prevWifiS = lastWifiS;
    lastWifiT = t; lastWifiS = steerValue;
    lastPcSentMs = mock::now_ms;
    pcSentEver = true;
    if (t == 0 && steerValue == 0) zeroPcSinceChange = true;
  }

  void handleWifiLine(const std::string& text) {
    std::istringstream in(text);
    std::string kind, item;
    in >> kind;
    std::map<std::string, std::string> fields;
    bool wellFormed = true;
    while (in >> item) {
      const size_t eq = item.find('=');
      if (eq == std::string::npos || eq == 0) { wellFormed = false; break; }
      fields[item.substr(0, eq)] = item.substr(eq + 1);
    }
    if (kind == "ST" && wellFormed && fields.size() == 13) { ++wifiStLines; wst = fields; }
    else if (kind == "NET" && wellFormed && fields.count("pc_via")) { ++wifiNetLines; wnet = fields; }
    else if (kind == "CH" && wellFormed && fields.count("arm")) { ++wifiChLines; }
    else if (kind == "HELLO" && wellFormed) { ++wifiHelloLines; wifiHello = text; }
    else { ++wifiBadLines; std::printf("  unexpected Wi-Fi line: \"%s\"\n", text.c_str()); }
  }

  void wifiStatus(const char* label) const {
    std::printf("  (%s, over Wi-Fi) armed=%s req=%s act=%s pc_ok=%s block=%s fault=%s stop=%s L=%s R=%s | sta=%s peer=%s pc_via=%s\n", label,
                ws("armed").c_str(), ws("req").c_str(), ws("act").c_str(), ws("pc_ok").c_str(), ws("block").c_str(),
                ws("fault").c_str(), ws("stop").c_str(), ws("L").c_str(), ws("R").c_str(),
                wn("sta").c_str(), wn("peer").c_str(), wn("pc_via").c_str());
  }

  void violation(const char* what) {
    if (++violations <= 5) std::printf("  invariant broken at %lu ms: %s\n", mock::now_ms, what);
  }

  void handleLine(const std::string& text) {
    longest = std::max(longest, static_cast<int>(text.size()) + 1);
    std::istringstream in(text);
    std::string kind, item;
    in >> kind;
    std::map<std::string, std::string> fields;
    bool wellFormed = true;
    while (in >> item) {
      const size_t eq = item.find('=');
      if (eq == std::string::npos || eq == 0) { wellFormed = false; break; }
      fields[item.substr(0, eq)] = item.substr(eq + 1);
    }
    static const char* stKeys[] = {"t", "tok", "link", "armed", "req", "act", "pc_ok", "block", "fault", "stop", "L", "R", "pc_age"};
    if (kind == "ST" && wellFormed && fields.size() == 13) {
      for (const char* key : stKeys) wellFormed = wellFormed && fields.count(key) == 1;
      if (!wellFormed) { ++badLines; return; }
      ++stLines;
      st = fields;
      seenAct += "," + fields["act"];
      // The line must agree with itself and with the pins at this instant.
      const Motors m = motors();
      const int l = std::atoi(fields["L"].c_str()), r = std::atoi(fields["R"].c_str());
      if (fields["act"] == "NONE" && (l != 0 || r != 0)) violation("ST: act=NONE with non-zero outputs");
      if (fields["pc_ok"] == "1" && fields["act"] != "PC") violation("ST: pc_ok=1 but act is not PC");
      if ((fields["block"] == "NONE") != (fields["act"] != "NONE")) violation("ST: block and act disagree");
      if (fields["armed"] == "0" && fields["act"] != "NONE") violation("ST: a source is active while disarmed");
      if (!m.valid || m.left != l || m.right != r) violation("ST: L/R differ from the motor pins");
    } else if (kind == "CH" && wellFormed && fields.count("arm") && fields.count("mode")) {
      ++chLines;
      ch = fields;
    } else if (kind == "HELLO" && wellFormed) {
      ++helloLines;
      hello = text;
    } else if (kind == "NET" && wellFormed && fields.count("pc_via")) {
      ++netLines;
      net = fields;
    } else {
      ++badLines;
      std::printf("  unexpected USB line: \"%s\"\n", text.c_str());
    }
  }

  void tick() {
    const unsigned long now = ++mock::now_ms;

    if (junk) {
      for (int i = 0; i < 6; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        ELRS.rx.push_back(static_cast<uint8_t>(lcg >> 24));
      }
      if (now % 50 == 0) pushFrame(ELRS.rx, 0x08, {0, 60, 0, 10, 0, 0, 50, 80});  // battery frame, valid CRC
    }
    if (radioOn && now % 4 == 0) {  // 250 Hz
      int frame[16];
      for (int& value : frame) value = 992;
      frame[STEER_CHANNEL - 1] = steer;
      frame[THROTTLE_CHANNEL - 1] = throttle;
      frame[ARM_CHANNEL - 1] = armRaw;
      frame[MODE_CHANNEL - 1] = modeRaw;
      pushFrame(ELRS.rx, 0x16, packChannels(frame));

      if (!everSent || now - lastFrameMs > 250) {
        framesSinceGap = 0;
        armOffSinceLink = false;
        neutralSinceChange = false;
        zeroPcSinceChange = false;
      }
      ++framesSinceGap;
      const Band armBand = band(armRaw), modeBand = band(modeRaw);
      const bool switchesMoved = !everSent || armBand != band(sent[ARM_CHANNEL - 1]) || modeBand != band(sent[MODE_CHANNEL - 1]);
      if (switchesMoved) {
        neutralSinceChange = false;
        zeroPcSinceChange = false;
      } else if (expectNorm(steer) == 0 && expectNorm(throttle) == 0) {
        neutralSinceChange = true;
      }
      if (armBand == (ARM_WHEN_HIGH ? BAND_LOW : BAND_HIGH) && framesSinceGap >= 10) armOffSinceLink = true;
      if (armBand == BAND_BAD) armOffSinceLink = false;
      std::memcpy(sent, frame, sizeof(sent));
      everSent = true;
      lastFrameMs = now;
    }
    if (statsOn && now % 100 == 0) pushFrame(ELRS.rx, 0x14, {60, 62, lq, 10, 0, 4, 3, 70, 100, 8});

    if (now % 20 == 0) {  // 50 Hz, like the PC client
      if (pc == PC_FOLLOW && st.count("tok")) sendDrive(token(), pcThrottle, pcSteer);
      else if (pc == PC_FIXED_TOKEN) sendDrive(fixedToken, pcThrottle, pcSteer);
      else if (pc == PC_SCRIPT && script) sendPc(script(*this));
    }

    if (now % 20 == 10) {  // the Wi-Fi laptop: also 50 Hz, between the USB client's sends
      if (wifi == WIFI_FOLLOW) {
        if (wst.count("tok")) sendWifiDrive(wifiToken(), wifiThrottle, wifiSteer);
        else sendWifiRaw("?\n");
      } else if (wifi == WIFI_FIXED_TOKEN) {
        sendWifiDrive(wifiFixedToken, wifiThrottle, wifiSteer);
      }
    }
    mock::stations = stationsForced >= 0 ? stationsForced : ((wifi != WIFI_OFF && wifiReachable) ? 1 : 0);

    loop();

    for (const mock::Datagram& datagram : mock::wifi_out) {
      ++wifiOutCount;
      if (!wifiReachable) continue;                         // nobody is listening
      if (datagram.ip != wifiIp || datagram.port != wifiPort) { ++wifiStray; continue; }
      wifiCapture += datagram.data;
      for (char c : datagram.data) {
        if (c == '\n') {
          handleWifiLine(wifiLine);
          wifiLine.clear();
        } else {
          wifiLine.push_back(c);
        }
      }
    }
    mock::wifi_out.clear();

    capture += HWCDCSerial.tx;
    for (char c : HWCDCSerial.tx) {
      if (c == '\n') {
        handleLine(line);
        line.clear();
      } else {
        line.push_back(c);
      }
    }
    HWCDCSerial.tx.clear();

    // ---- invariants, every millisecond ----
    const Motors m = motors();
    if (!m.valid) violation("motor pins are not brake, forward or reverse");
    if (m.valid && !m.stopped()) {
      if (!everSent || now - lastFrameMs > 250) violation("driving with no radio frame in the last 250 ms");
      if (band(sent[ARM_CHANNEL - 1]) != (ARM_WHEN_HIGH ? BAND_HIGH : BAND_LOW)) violation("driving with the ARM switch not on");
      if (!armOffSinceLink) violation("driving without ARM having been seen off on this link");
      const Band modeBand = band(sent[MODE_CHANNEL - 1]);
      int l = 0, r = 0;
      if (modeBand == BAND_BAD) {
        violation("driving with an invalid MODE switch");
      } else if (modeBand == (PC_WHEN_HIGH ? BAND_HIGH : BAND_LOW)) {
        if (!zeroPcSinceChange) violation("PC drive without a zero command since the last switch change");
        if (!pcSentEver || now - lastPcSentMs > 250) violation("PC drive with no PC command in the last 250 ms");
        expectMix(constrain(lastPcT, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), constrain(lastPcS, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), l, r);
        bool match = m.left == l && m.right == r;
        expectMix(constrain(prevPcT, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), constrain(prevPcS, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), l, r);
        match = match || (m.left == l && m.right == r);
        expectMix(constrain(lastWifiT, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), constrain(lastWifiS, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), l, r);
        match = match || (m.left == l && m.right == r);
        expectMix(constrain(prevWifiT, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), constrain(prevWifiS, -PC_COMMAND_LIMIT, PC_COMMAND_LIMIT), l, r);
        match = match || (m.left == l && m.right == r);
        if (!match) violation("PC drive does not match a recent PC command");
      } else {
        if (!neutralSinceChange) violation("RADIO drive without neutral sticks since the last switch change");
        expectMix(expectNorm(sent[THROTTLE_CHANNEL - 1]), expectNorm(sent[STEER_CHANNEL - 1]), l, r);
        if (m.left != l || m.right != r) violation("RADIO drive does not match the sticks");
      }
    }
  }

  void run(int ms) {
    for (int i = 0; i < ms; ++i) tick();
  }

  // True if the motors stayed braked for the whole period.
  bool stoppedFor(int ms) {
    bool all = true;
    for (int i = 0; i < ms; ++i) {
      tick();
      all = all && motors().stopped();
    }
    return all;
  }

  // True if a wheel was driven on every tick of the period.
  bool movingFor(int ms) {
    bool all = true;
    for (int i = 0; i < ms; ++i) {
      tick();
      all = all && !motors().stopped();
    }
    return all;
  }

  // Ticks until both motors are braked, or -1.
  int msUntilStopped(int limit) {
    for (int i = 0; i <= limit; ++i) {
      if (motors().stopped()) return i;
      tick();
    }
    return -1;
  }

  // Run up to and including the tick that sends a radio frame.
  void runToFrame() {
    do tick(); while (mock::now_ms % 4 != 0);
  }
  // Run up to and including the tick that sends a PC command.
  void runToPcSend() {
    do tick(); while (mock::now_ms % 20 != 0);
  }

  void boot() {
    mock::on_delay = [](unsigned long) {
      check(motors().stopped(), "setup(): all four driver inputs are HIGH (brake) before the first delay");
    };
    setup();
    mock::on_delay = nullptr;
    check(ELRS.begun_baud == 420000 && ELRS.begun_rx == 20 && ELRS.begun_tx == -1 && ELRS.number == 1,
          "receiver UART: UART1, 420000 baud, RX on GPIO20, no TX pin");
    check(AIN1 == 3 && AIN2 == 2 && BIN1 == 4 && BIN2 == 5, "motor pins: AIN1/AIN2 = GPIO3/GPIO2, BIN1/BIN2 = GPIO4/GPIO5");
  }

  // Radio up, ARM off for a while, then ARM on with neutral sticks, in RADIO.
  bool armInRadio() {
    radioOn = true;
    armRaw = ARM_OFF; modeRaw = MODE_RADIO; steer = throttle = 992;
    run(120);
    armRaw = ARM_ON;
    run(60);
    return is("armed", "1") && is("act", "RADIO") && is("block", "NONE") && motors().stopped();
  }

  // Radio up, ARM off, then ARM on in PC with a client sending zero.
  bool armInPc() {
    radioOn = true;
    armRaw = ARM_OFF; modeRaw = MODE_PC; steer = throttle = 992;
    pc = PC_FOLLOW; pcThrottle = pcSteer = 0;
    run(120);
    armRaw = ARM_ON;
    run(120);
    return is("armed", "1") && is("act", "PC") && is("pc_ok", "1") && motors().stopped();
  }

  // Run up to and including the tick in which the Wi-Fi laptop sends.
  void runToWifiSend() {
    do tick(); while (mock::now_ms % 20 != 10);
  }

  // Radio up, ARM off, then ARM on in PC with the Wi-Fi laptop sending zero.
  bool armInPcWifi() {
    radioOn = true;
    armRaw = ARM_OFF; modeRaw = MODE_PC; steer = throttle = 992;
    wifi = WIFI_FOLLOW; wifiThrottle = wifiSteer = 0;
    run(150);
    armRaw = ARM_ON;
    run(150);
    return wis("armed", "1") && wis("act", "PC") && wis("pc_ok", "1") && wn("pc_via") == "WIFI" && motors().stopped();
  }

  void status(const char* label) const {
    std::printf("  (%s) armed=%s req=%s act=%s pc_ok=%s block=%s fault=%s stop=%s L=%s R=%s\n", label,
                s("armed").c_str(), s("req").c_str(), s("act").c_str(), s("pc_ok").c_str(), s("block").c_str(),
                s("fault").c_str(), s("stop").c_str(), s("L").c_str(), s("R").c_str());
  }
};

static Rig rig;

// ---------------------------------------------------------------- scenarios

static void boot_no_radio() {
  rig.boot();
  check(mock::digital_writes == 0, "setup() never calls digitalWrite (%lu calls)", mock::digital_writes);
  check(rig.stoppedFor(1000), "no radio for 1 s: braked throughout");
  rig.status("no radio");
  check(rig.is("block", "NO_RADIO") && rig.is("armed", "0") && rig.is("link", "0") && rig.is("act", "NONE"),
        "status: link=0 armed=0 act=NONE block=NO_RADIO");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 100;
  check(rig.stoppedFor(1000), "PC sends forward with the live token and no radio: braked throughout");
  check(rig.is("armed", "0") && std::atoi(rig.s("pc_age").c_str()) >= 0 && std::atoi(rig.s("pc_age").c_str()) <= 20,
        "status still armed=0, and pc_age shows the PC client is talking (%s ms)", rig.s("pc_age").c_str());
  check(rig.helloLines == 1 && rig.hello.find("arm=CH5:HIGH") != std::string::npos && rig.hello.find("pc=CH6:HIGH") != std::string::npos,
        "HELLO line names the configured ARM and MODE channels: %s", rig.hello.c_str());
  rig.sendPc("?\n");
  rig.run(600);
  check(rig.helloLines == 2, "\"?\" from the PC gets the HELLO line again");
}

static void boot_arm_already_on() {
  rig.boot();
  rig.radioOn = true;
  rig.armRaw = ARM_ON; rig.modeRaw = MODE_RADIO;
  check(rig.stoppedFor(500), "RADIO, ARM already on at power-up, sticks neutral: braked for 0.5 s");
  rig.status("ARM on at boot");
  check(rig.is("block", "ARM_ON_AT_BOOT") && rig.is("armed", "0") && rig.is("link", "1"), "status: link=1 armed=0 block=ARM_ON_AT_BOOT");
  rig.throttle = 1811;
  check(rig.stoppedFor(1000), "... full throttle for 1 s: still braked");
  rig.throttle = 992;
  check(rig.stoppedFor(500), "... neutral again: still braked (neutral alone does not arm)");
  rig.throttle = 1811;
  rig.armRaw = ARM_OFF;
  rig.run(100);
  check(rig.is("block", "DISARMED"), "ARM off: block=DISARMED");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(1000), "ARM on with the stick still at full throttle: braked for 1 s");
  rig.status("armed, stick held");
  check(rig.is("armed", "1") && rig.is("block", "WAIT_RADIO_NEUTRAL") && rig.is("act", "NONE"), "status: armed=1 act=NONE block=WAIT_RADIO_NEUTRAL");
  rig.throttle = 992;
  rig.run(20);
  check(rig.is("act", "RADIO") && rig.is("block", "NONE") && motors().stopped(), "stick to neutral: act=RADIO, still braked");
  rig.throttle = 1811;
  rig.run(20);
  check(motors().left == 255 && motors().right == 255, "full throttle now drives: L=%d R=%d", motors().left, motors().right);
}

static void boot_arm_already_on_pc() {
  rig.boot();
  rig.radioOn = true;
  rig.armRaw = ARM_ON; rig.modeRaw = MODE_PC;
  rig.pc = PC_FOLLOW;
  check(rig.stoppedFor(500), "PC selected, ARM already on at power-up, PC sending zero: braked");
  rig.pcThrottle = 100;
  check(rig.stoppedFor(1000), "... PC sends forward for 1 s: still braked");
  check(rig.is("block", "ARM_ON_AT_BOOT") && rig.is("armed", "0") && rig.is("pc_ok", "0") && rig.is("req", "PC"),
        "status: armed=0 req=PC pc_ok=0 block=ARM_ON_AT_BOOT");
  rig.pcThrottle = 0;
  check(rig.stoppedFor(500), "... PC back to zero: still braked (the PC cannot arm)");
  check(rig.is("armed", "0"), "status: still armed=0");
}

static void drive_matches_original() {
  static const int values[] = {0, 100, 174, 175, 300, 500, 700, 900, 946, 947, 948, 991, 992, 993,
                               1036, 1037, 1038, 1100, 1300, 1500, 1700, 1810, 1811, 1812, 1900, 2047};
  const int count = static_cast<int>(sizeof(values) / sizeof(values[0]));
  std::vector<std::pair<int, int>> want;

  // ---- the original, untouched, on the same mock ----
  original::setup();
  auto feedOriginal = [](int ch1, int ch2) {
    int frame[16];
    for (int& value : frame) value = 992;
    frame[0] = ch1; frame[1] = ch2;
    pushFrame(original::ELRS.rx, 0x16, packChannels(frame));
  };
  for (int i = 0; i < 200; ++i) {
    ++mock::now_ms;
    if (mock::now_ms % 4 == 0) feedOriginal(992, 992);
    original::loop();
  }
  check(HWCDCSerial.tx.find("RC CONTROL ENABLED") != std::string::npos, "original sketch: enabled after 10 frames with centred sticks");
  int unread = 0;
  for (int a = 0; a < count; ++a) {
    for (int b = 0; b < count; ++b) {
      HWCDCSerial.tx.clear();
      int left = 9999, right = 9999;
      char prefix[48];
      std::snprintf(prefix, sizeof(prefix), "CH1:%d CH2:%d |", values[a], values[b]);
      for (int i = 0; i < 400 && left == 9999; ++i) {
        ++mock::now_ms;
        if (mock::now_ms % 4 == 0) feedOriginal(values[a], values[b]);
        original::loop();
        const size_t at = HWCDCSerial.tx.rfind(prefix);
        if (at != std::string::npos && HWCDCSerial.tx.find('\n', at) != std::string::npos) {
          const size_t l = HWCDCSerial.tx.find(" L:", at), r = HWCDCSerial.tx.find(" R:", at);
          left = std::atoi(HWCDCSerial.tx.c_str() + l + 3);
          right = std::atoi(HWCDCSerial.tx.c_str() + r + 3);
        }
      }
      if (left == 9999) ++unread;
      want.push_back({left, right});
    }
  }
  check(unread == 0, "original sketch: read its own L/R for all %d stick positions", count * count);

  // ---- the new sketch, RADIO mode ----
  HWCDCSerial.tx.clear();
  mock::resetBoard();
  rig.boot();
  check(rig.armInRadio(), "new sketch: armed in RADIO with neutral sticks");
  int pinMismatch = 0, valueMismatch = 0, formulaMismatch = 0, shown = 0;
  for (int a = 0; a < count; ++a) {
    for (int b = 0; b < count; ++b) {
      rig.steer = values[a]; rig.throttle = values[b];
      rig.run(8);
      const Motors m = motors();
      const std::pair<int, int>& w = want[a * count + b];
      int l, r;
      expectMix(expectNorm(values[b]), expectNorm(values[a]), l, r);
      if (l != w.first || r != w.second) ++formulaMismatch;
      if (outLeft != w.first || outRight != w.second) ++valueMismatch;
      const bool pinsOk = m.valid && m.left == w.first && m.right == w.second &&
                          m.leftBrake == (w.first == 0) && m.rightBrake == (w.second == 0);
      if (!pinsOk) {
        ++pinMismatch;
        if (shown++ < 5) std::printf("  CH1=%d CH2=%d: original L=%d R=%d, new pins L=%d R=%d\n", values[a], values[b], w.first, w.second, m.left, m.right);
      }
    }
  }
  check(valueMismatch == 0, "new sketch computes the same L/R as the original at all %d stick positions", count * count);
  check(pinMismatch == 0, "new sketch's motor pins carry those L/R (forward/reverse side, duty, brake at 0)");
  check(formulaMismatch == 0, "the test's own calibration + mixing formula agrees with the original too");

  // Wheel polarity, pin by pin.
  rig.steer = 992; rig.throttle = 1500; rig.run(8);
  check(mock::pinOut(AIN1) > 0 && mock::pinOut(AIN1) < 256 && mock::pinOut(AIN2) == 0 && mock::pinOut(BIN1) == 0 && mock::pinOut(BIN2) == mock::pinOut(AIN1),
        "forward: AIN1 PWM, AIN2 LOW, BIN1 LOW, BIN2 PWM (AIN1 %s, BIN2 %s)", pinText(AIN1).c_str(), pinText(BIN2).c_str());
  rig.throttle = 500; rig.run(8);
  check(mock::pinOut(AIN1) == 0 && mock::pinOut(AIN2) > 0 && mock::pinOut(BIN2) == 0 && mock::pinOut(BIN1) == mock::pinOut(AIN2),
        "reverse: AIN1 LOW, AIN2 PWM, BIN1 PWM, BIN2 LOW");
  rig.throttle = 992; rig.steer = 1500; rig.run(8);
  check(motors().left > 0 && motors().right == -motors().left, "stick right: left wheel forward, right wheel reverse");

  // Same bytes into both parsers.
  int decodeMismatch = 0;
  uint32_t seed = 99;
  for (int n = 0; n < 300; ++n) {
    int frame[16];
    for (int& value : frame) {
      seed = seed * 1664525u + 1013904223u;
      value = static_cast<int>((seed >> 16) % 2048);
    }
    frame[ARM_CHANNEL - 1] = ARM_OFF;
    pushFrame(original::ELRS.rx, 0x16, packChannels(frame));
    pushFrame(ELRS.rx, 0x16, packChannels(frame));
    original::readCRSF();
    readCRSF();
    for (int c = 0; c < 16; ++c) {
      if (original::channels[c] != channels[c] || channels[c] != frame[c]) ++decodeMismatch;
    }
  }
  check(decodeMismatch == 0, "300 random RC frames decode to the same 16 channels in both sketches");
}

static void stop_path() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  auto allHigh = [] {
    return mock::pinOut(AIN1) == 256 && mock::pinOut(AIN2) == 256 && mock::pinOut(BIN1) == 256 && mock::pinOut(BIN2) == 256;
  };
  auto ramp = [](int from, int to) {  // a stick moved by hand, 100 ms
    for (int i = 1; i <= 25; ++i) {
      rig.throttle = from + (to - from) * i / 25;
      rig.run(4);
    }
  };
  ramp(992, 1811);
  check(motors().left == 255 && motors().right == 255, "forward: L=R=255");
  ramp(1811, 992);
  rig.run(8);
  check(allHigh(), "stick released after forward: GPIO3, GPIO2, GPIO4, GPIO5 all constant HIGH (%s %s %s %s)",
        pinText(AIN1).c_str(), pinText(AIN2).c_str(), pinText(BIN1).c_str(), pinText(BIN2).c_str());
  ramp(992, 174);
  check(motors().left == -255 && motors().right == -255, "reverse: L=R=-255");
  ramp(174, 992);
  rig.run(8);
  check(allHigh(), "stick released after reverse: all four constant HIGH");
  rig.steer = 1811; rig.run(40);
  rig.steer = 992; rig.run(8);
  check(allHigh(), "stick released after a spin: all four constant HIGH");
  rig.throttle = 1811; rig.run(40);
  rig.armRaw = ARM_OFF; rig.run(8);
  check(allHigh(), "ARM off at full throttle: all four constant HIGH");
  rig.throttle = 992; rig.run(100); rig.armRaw = ARM_ON; rig.run(40);
  rig.throttle = 174; rig.run(40);
  rig.radioOn = false; rig.run(300);
  check(allHigh(), "transmitter off at full reverse: all four constant HIGH");
  check(mock::digital_writes == 0, "digitalWrite is never called on any pin (%lu calls)", mock::digital_writes);
}

// Report only: what the core's source says the ORIGINAL sketch does at a stop.
static void original_stop_path_model() {
  original::setup();
  int throttle = 992;
  bool on = true;
  auto runOriginal = [&](int ms) {
    for (int i = 0; i < ms; ++i) {
      ++mock::now_ms;
      if (on && mock::now_ms % 4 == 0) {
        int frame[16];
        for (int& value : frame) value = 992;
        frame[1] = throttle;
        pushFrame(original::ELRS.rx, 0x16, packChannels(frame));
      }
      original::loop();
    }
  };
  auto ramp = [&](int from, int to) {
    for (int i = 1; i <= 25; ++i) {
      throttle = from + (to - from) * i / 25;
      runOriginal(4);
    }
  };
  auto row = [](const char* step) {
    std::printf("  %-44s | %-11s | %-11s | %-11s | %-11s\n", step, pinText(AIN1).c_str(), pinText(AIN2).c_str(), pinText(BIN1).c_str(), pinText(BIN2).c_str());
  };
  std::printf("  MODEL of ELRS-Control.ino on core 3.3.0 (digitalWrite ignored on a PWM pin). Not a measurement.\n");
  std::printf("  %-44s | %-11s | %-11s | %-11s | %-11s\n", "step", "AIN1 GPIO3", "AIN2 GPIO2", "BIN1 GPIO4", "BIN2 GPIO5");
  runOriginal(200); row("after start-up, sticks never moved");
  ramp(992, 1811); row("full forward");
  ramp(1811, 992); runOriginal(20); row("stick released (only forward used so far)");
  ramp(992, 174); row("full reverse");
  ramp(174, 992); runOriginal(20); row("stick released (both directions used)");
  ramp(992, 1811); row("full forward again");
  on = false; runOriginal(400); row("transmitter off at full forward");
  check(mock::ignored_digital_writes > 0, "model: the original's digitalWrite calls on PWM pins were ignored (%lu of %lu)",
        mock::ignored_digital_writes, mock::digital_writes);
}

static void disarm_radio() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.throttle = 1811; rig.run(40);
  check(motors().left == 255, "driving forward");
  rig.runToFrame();
  rig.armRaw = ARM_OFF;
  const int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "ARM off: braked %d ms later (one radio frame is 4 ms)", ms);
  rig.run(40);
  rig.status("disarmed");
  check(rig.is("armed", "0") && rig.is("block", "DISARMED") && rig.is("stop", "DISARM") && rig.is("fault", "NONE"),
        "status: armed=0 block=DISARMED stop=DISARM fault=NONE");
  check(rig.stoppedFor(1000), "stick still at full throttle for 1 s: braked");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(1000), "ARM on again with the stick held: braked for 1 s");
  check(rig.is("block", "WAIT_RADIO_NEUTRAL"), "status: block=WAIT_RADIO_NEUTRAL");
  rig.throttle = 992; rig.run(20);
  rig.throttle = 1811; rig.run(20);
  check(motors().left == 255, "neutral, then throttle: drives again");
}

static void disarm_pc() {
  rig.boot();
  check(rig.armInPc(), "armed in PC, client sending zero: pc_ok=1");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "PC forward 100: L=R=100");
  rig.runToFrame();
  rig.armRaw = ARM_OFF;
  const int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "ARM off in PC mode: braked %d ms later", ms);
  check(rig.stoppedFor(1000), "PC keeps sending forward (key held) for 1 s: braked");
  check(rig.is("armed", "0") && rig.is("pc_ok", "0") && rig.is("stop", "DISARM"), "status: armed=0 pc_ok=0 stop=DISARM");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(1000), "ARM on again, PC still sending forward with the new token: braked for 1 s");
  rig.status("re-armed, key held");
  check(rig.is("armed", "1") && rig.is("block", "WAIT_PC_NEUTRAL") && rig.is("pc_ok", "0"), "status: armed=1 pc_ok=0 block=WAIT_PC_NEUTRAL");
  rig.pcThrottle = 0; rig.run(60);
  check(rig.is("pc_ok", "1") && motors().stopped(), "PC sends zero: pc_ok=1, still braked");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC forward: drives again");
}

static void handoff_radio_to_pc() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 100;  // PC key held the whole time
  rig.throttle = 1811; rig.run(60);
  check(motors().left == 255 && motors().right == 255, "RADIO full forward while the PC sends forward 100: L=R=255 (PC ignored)");
  const unsigned long before = rig.token();
  rig.runToFrame();
  rig.modeRaw = MODE_PC;
  const int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "MODE to PC with stick and key both held: braked %d ms later", ms);
  check(rig.stoppedFor(1000), "... braked for the next 1 s");
  rig.status("handoff to PC, both held");
  check(rig.is("req", "PC") && rig.is("act", "NONE") && rig.is("pc_ok", "0") && rig.is("block", "WAIT_PC_NEUTRAL") && rig.is("stop", "HANDOFF") && rig.is("armed", "1"),
        "status: req=PC act=NONE pc_ok=0 block=WAIT_PC_NEUTRAL stop=HANDOFF armed=1");
  check(rig.token() != before, "the token changed at the handoff");
  rig.pcThrottle = 0; rig.run(60);
  check(rig.is("act", "PC") && rig.is("pc_ok", "1") && motors().stopped(), "PC key released (zero command): act=PC pc_ok=1, braked");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "PC forward 100: L=R=100, with the radio stick still at full");
  rig.pcThrottle = 0; rig.pcSteer = 60; rig.run(60);
  check(motors().left == 60 && motors().right == -60, "PC steer right 60: L=60 R=-60");
}

static void handoff_pc_to_radio() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.throttle = 1811;  // radio stick held the whole time
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "PC forward 100 while the radio stick is at full: L=R=100 (stick ignored)");
  rig.runToFrame();
  rig.modeRaw = MODE_RADIO;
  const int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "MODE to RADIO with stick and key both held: braked %d ms later", ms);
  check(rig.stoppedFor(1000), "... braked for the next 1 s");
  rig.status("handoff to RADIO, both held");
  check(rig.is("req", "RADIO") && rig.is("act", "NONE") && rig.is("block", "WAIT_RADIO_NEUTRAL") && rig.is("stop", "HANDOFF") && rig.is("pc_ok", "0"),
        "status: req=RADIO act=NONE pc_ok=0 block=WAIT_RADIO_NEUTRAL stop=HANDOFF");
  rig.throttle = 1045; rig.run(100);
  check(expectNorm(1045) == 2 && motors().stopped() && rig.is("block", "WAIT_RADIO_NEUTRAL"),
        "stick at 1045 (inside the original's +/-70 'centred' window, but it normalises to 2, not 0): still waiting");
  rig.throttle = 992; rig.run(20);
  check(rig.is("act", "RADIO") && motors().stopped(), "stick to true neutral: act=RADIO, braked");
  rig.throttle = 1500; rig.run(20);
  int l, r;
  expectMix(expectNorm(1500), 0, l, r);
  check(motors().left == l && motors().right == r && l > 0, "stick forward: L=R=%d from the stick, PC still sending 100 and ignored", l);
}

static void handoff_flapping() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 100;
  rig.throttle = 1811;
  bool stopped = true;
  rig.modeRaw = MODE_PC; rig.run(8);
  for (int i = 0; i < 500; ++i) {  // MODE flipped every 8 ms for 4 s, stick and key held
    rig.modeRaw = (i % 2) ? MODE_PC : MODE_RADIO;
    stopped = rig.stoppedFor(8) && stopped;
  }
  check(stopped, "MODE flipped every 8 ms for 4 s with stick and key held: never drives");
  int alternating = 0;
  rig.throttle = 992;
  for (int i = 0; i < 100; ++i) {  // PC alternates zero / forward while the switch flaps
    rig.modeRaw = (i % 2) ? MODE_PC : MODE_RADIO;
    rig.pcThrottle = (i % 3) ? 100 : 0;
    rig.run(40);
    ++alternating;
  }
  check(rig.violations == 0, "%d more flips with the PC alternating zero and forward: every drive tick had a fresh neutral first", alternating);
}

static void pc_stale() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.throttle = 1811;  // radio stick held at full throughout
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC driving forward 100");
  rig.runToPcSend();    // the last command the PC ever sends
  rig.pc = PC_SILENT;
  check(rig.movingFor(249), "PC goes silent: still driving 249 ms after the last command");
  rig.tick();
  check(motors().stopped(), "... braked at 250 ms");
  rig.run(40);
  rig.seenAct.clear();
  rig.status("PC stale");
  check(rig.is("armed", "0") && rig.is("act", "NONE") && rig.is("block", "REARM_REQUIRED") && rig.is("fault", "PC_STALE") && rig.is("stop", "PC_STALE"),
        "status: armed=0 act=NONE block=REARM_REQUIRED fault=PC_STALE stop=PC_STALE");
  check(rig.stoppedFor(1000), "radio stick at full the whole time: braked for 1 s (no fall-back to RADIO)");
  rig.modeRaw = MODE_RADIO;
  check(rig.stoppedFor(500), "MODE to RADIO without re-arming, stick at full: braked");
  rig.throttle = 992;
  check(rig.stoppedFor(500), "... stick to neutral: still braked");
  rig.throttle = 1811;
  check(rig.stoppedFor(500) && rig.is("block", "REARM_REQUIRED"), "... stick forward: still braked, block=REARM_REQUIRED");
  check(rig.seenAct.find("RADIO") == std::string::npos && rig.seenAct.find("PC") == std::string::npos, "act was never RADIO or PC after the stale stop");
  rig.modeRaw = MODE_PC;
  rig.pc = PC_FOLLOW; rig.pcThrottle = 0;
  check(rig.stoppedFor(300), "PC client comes back sending zero: braked");
  rig.pcThrottle = 100;
  check(rig.stoppedFor(700) && rig.is("pc_ok", "0"), "PC sends forward: braked, pc_ok=0 (the PC cannot clear the fault)");
  rig.throttle = 992;
  rig.armRaw = ARM_OFF; rig.run(100);
  check(rig.is("fault", "NONE") && rig.is("block", "DISARMED"), "ARM off: fault cleared, block=DISARMED");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.is("block", "WAIT_PC_NEUTRAL"), "ARM on, PC still sending forward: braked, block=WAIT_PC_NEUTRAL");
  rig.pcThrottle = 0; rig.run(60);
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC zero, then forward: drives again");
}

static void pc_garbage_is_not_alive() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC driving forward 100");
  rig.runToPcSend();
  int n = 0;
  rig.pc = PC_SCRIPT;
  rig.script = [&n](Rig& r) {
    char text[200];
    const unsigned long tok = r.token();
    switch (n++ % 12) {
      case 0: std::snprintf(text, sizeof(text), "D,%lu,100\n", tok); break;          // field missing
      case 1: std::snprintf(text, sizeof(text), "D,%lu,300,0\n", tok); break;        // out of range
      case 2: std::snprintf(text, sizeof(text), "D,%lu,100,0\n", tok + 1); break;    // wrong token
      case 3: std::snprintf(text, sizeof(text), "X,%lu,100,0\n", tok); break;        // wrong letter
      case 4: std::snprintf(text, sizeof(text), "D,%lu,100,0,5\n", tok); break;      // extra field
      case 5: std::snprintf(text, sizeof(text), "D,%lu, 100,0\n", tok); break;       // space
      case 6: std::snprintf(text, sizeof(text), "D,-1,100,0\n"); break;              // negative token
      case 7: std::snprintf(text, sizeof(text), "D,%lu,1e2,0\n", tok); break;        // not an integer
      case 8: std::snprintf(text, sizeof(text), "D,%lu,100,0\r\n", tok); break;      // CR LF
      case 9: std::snprintf(text, sizeof(text), "D,99999999999,100,0\n"); break;     // token overflow
      case 10: std::snprintf(text, sizeof(text), "D,%lu,100,0,00000000000000000000000000000000000000000000\n", tok); break;  // too long
      default: std::snprintf(text, sizeof(text), "d,%lu,100,0\n", tok); break;       // lower case
    }
    return std::string(text);
  };
  const int ms = rig.msUntilStopped(400);
  check(ms >= 230 && ms <= 250, "only malformed and wrong-token lines from here: braked %d ms after the last good command", ms);
  rig.run(1100);
  check(rig.is("fault", "PC_STALE"), "status: fault=PC_STALE");
  check(std::atoi(rig.ch["pc_bad"].c_str()) > 0, "CH line counts the malformed lines (pc_bad=%s)", rig.ch["pc_bad"].c_str());
}

static void pc_reconnect_old_command() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.pcThrottle = 100; rig.run(60);
  const unsigned long oldToken = rig.token();
  check(motors().left == 100, "PC driving forward 100, token %lu", oldToken);
  rig.runToPcSend();
  rig.pc = PC_SILENT;                       // client killed / USB pulled
  rig.run(400);
  check(motors().stopped() && rig.is("fault", "PC_STALE"), "client gone: braked, fault=PC_STALE");
  rig.pc = PC_FIXED_TOKEN; rig.fixedToken = oldToken; rig.pcThrottle = 100;
  check(rig.stoppedFor(1000), "client back, replaying the old command with the old token: braked for 1 s");
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(1000), "radio re-armed, old-token forward still arriving: braked for 1 s");
  rig.pcThrottle = 0;
  check(rig.stoppedFor(300), "old-token zero: braked");
  rig.pcThrottle = 100;
  check(rig.stoppedFor(700) && rig.is("pc_ok", "0") && rig.is("block", "WAIT_PC_NEUTRAL"),
        "old-token forward after old-token zero: braked, pc_ok=0, block=WAIT_PC_NEUTRAL");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 100;
  check(rig.stoppedFor(1000) && rig.is("pc_ok", "0"), "current token but the key is still held (forward): braked for 1 s");
  rig.pcThrottle = 0; rig.run(60);
  check(rig.is("pc_ok", "1") && motors().stopped(), "current token, zero: pc_ok=1");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "then forward: drives");
}

static void radio_loss_in_radio() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 0;   // a PC client is connected and idle
  rig.throttle = 1811; rig.run(60);
  check(motors().left == 255, "driving full forward");
  rig.runToFrame();
  rig.radioOn = false;                      // transmitter off, stick was forward
  rig.seenAct.clear();
  check(rig.movingFor(250), "transmitter off: still driving 250 ms after the last frame");
  rig.tick();
  check(motors().stopped(), "... braked at 251 ms");
  rig.run(40);
  rig.status("radio lost");
  check(rig.is("link", "0") && rig.is("armed", "0") && rig.is("block", "RADIO_LOST") && rig.is("fault", "RADIO_LOST") && rig.is("stop", "RADIO_LOST"),
        "status: link=0 armed=0 block=RADIO_LOST fault=RADIO_LOST stop=RADIO_LOST");
  rig.pcThrottle = 100;
  check(rig.stoppedFor(1000), "PC sends forward while the radio is gone: braked (no fall-back to PC)");
  rig.pcThrottle = 0;
  rig.radioOn = true;                       // link returns, ARM still on, stick still forward
  check(rig.stoppedFor(1000), "transmitter back with ARM on and the stick forward: braked for 1 s");
  rig.status("radio back");
  check(rig.is("link", "1") && rig.is("armed", "0") && rig.is("block", "REARM_REQUIRED") && rig.is("fault", "RADIO_LOST"),
        "status: link=1 armed=0 block=REARM_REQUIRED fault=RADIO_LOST");
  rig.throttle = 992;
  check(rig.stoppedFor(500) && rig.is("armed", "0"), "stick to neutral: still disarmed (neutral is not a re-arm)");
  rig.throttle = 1811;
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.is("block", "WAIT_RADIO_NEUTRAL"), "ARM off then on with the stick forward: braked, block=WAIT_RADIO_NEUTRAL");
  rig.throttle = 992; rig.run(20);
  rig.throttle = 1811; rig.run(20);
  check(motors().left == 255, "neutral, then throttle: drives again");
  check(rig.seenAct.find("PC") == std::string::npos, "act was never PC");
}

static void radio_loss_in_pc() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC driving forward 100");
  rig.runToFrame();
  rig.radioOn = false;                      // transmitter off; the PC keeps sending forward
  rig.seenAct.clear();
  check(rig.movingFor(250), "transmitter off in PC mode: still driving 250 ms after the last frame");
  rig.tick();
  check(motors().stopped(), "... braked at 251 ms although PC commands keep arriving");
  check(rig.stoppedFor(1000), "PC keeps sending forward with the live token for 1 s: braked");
  rig.status("radio lost in PC");
  check(rig.is("armed", "0") && rig.is("pc_ok", "0") && rig.is("block", "RADIO_LOST") && rig.is("stop", "RADIO_LOST"),
        "status: armed=0 pc_ok=0 block=RADIO_LOST stop=RADIO_LOST");
  rig.radioOn = true;
  check(rig.stoppedFor(1000), "transmitter back (ARM on, MODE PC), PC still sending forward: braked for 1 s");
  check(rig.is("block", "REARM_REQUIRED") && rig.is("fault", "RADIO_LOST"), "status: block=REARM_REQUIRED fault=RADIO_LOST");
  rig.pcThrottle = 0; rig.run(100);
  rig.pcThrottle = 100;
  check(rig.stoppedFor(500) && rig.is("pc_ok", "0"), "PC zero then forward without a re-arm: braked, pc_ok=0");
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.is("block", "WAIT_PC_NEUTRAL"), "ARM off then on, PC key still held: braked, block=WAIT_PC_NEUTRAL");
  rig.pcThrottle = 0; rig.run(60);
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100, "PC zero, then forward: drives again");
  check(rig.seenAct.find("RADIO") == std::string::npos, "act was never RADIO");
}

static void radio_reports_zero_link_quality() {
  rig.boot();
  rig.statsOn = true;
  check(rig.armInRadio(), "armed in RADIO, receiver reporting 100 %% link quality");
  rig.throttle = 1811; rig.run(60);
  check(motors().left == 255, "driving full forward");
  rig.lq = 0;                               // receiver says the link is gone but keeps sending frames
  const int ms = rig.msUntilStopped(400);
  check(ms >= 0 && ms <= 100, "receiver reports 0 %% while RC frames keep coming: braked %d ms later (reports are 100 ms apart)", ms);
  rig.run(40);
  check(rig.is("block", "RADIO_LOST") && rig.is("fault", "RADIO_LOST"), "status: block=RADIO_LOST fault=RADIO_LOST");
  rig.lq = 100; rig.run(300);
  check(motors().stopped() && rig.is("block", "REARM_REQUIRED"), "report back to 100 %%: still braked, block=REARM_REQUIRED");
}

static void radio_noise_is_not_a_link() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.throttle = 1811; rig.run(60);
  rig.runToFrame();
  rig.radioOn = false;
  rig.junk = true;                          // random bytes plus valid non-RC frames
  const int ms = rig.msUntilStopped(600);
  check(ms >= 247 && ms <= 251, "RC frames replaced by noise and battery frames: braked %d ms after the last RC frame", ms);
  check(rig.stoppedFor(2000) && rig.is("link", "0"), "2 s of noise: braked, link=0");
}

static void short_rc_frame() {
  rig.boot();
  // Type 0x16 with a valid CRC but the wrong length. Decoding 22 bytes from it
  // (as the original does) would read ARM on, MODE PC and full throttle.
  int frame[16];
  for (int& value : frame) value = 1811;
  frame[ARM_CHANNEL - 1] = ARM_ON; frame[MODE_CHANNEL - 1] = MODE_PC;
  std::vector<uint8_t> payload = packChannels(frame);
  std::vector<uint8_t> shortPayload(payload.begin(), payload.begin() + 12);
  std::vector<uint8_t> longPayload = payload;
  longPayload.push_back(0x55); longPayload.push_back(0x55);
  for (int i = 0; i < 300; ++i) {
    pushFrame(ELRS.rx, 0x16, (i % 2) ? shortPayload : longPayload);
    rig.tick();
  }
  rig.run(300);
  bool untouched = true;
  for (int c = 0; c < 16; ++c) untouched = untouched && channels[c] == 0;
  check(untouched && goodFrames == 0 && badLength == 300, "300 RC frames of the wrong length (valid CRC): none decoded, len counter = %lu", (unsigned long)badLength);
  check(rig.is("link", "0") && rig.is("block", "NO_RADIO") && motors().stopped(), "they do not qualify the link: link=0 block=NO_RADIO");
  check(rig.armInRadio(), "then a healthy stream: arms in RADIO as usual");
  rig.throttle = 1811; rig.run(40);
  for (int i = 0; i < 50; ++i) {
    pushFrame(ELRS.rx, 0x16, shortPayload);
    rig.tick();
  }
  check(motors().left == 255 && rig.violations == 0, "wrong-length frames mixed into a live stream change nothing");
}

static void switch_invalid() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.throttle = 1811; rig.run(40);
  rig.runToFrame();
  rig.armRaw = 992;                         // three-position switch in the middle
  int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "ARM channel at mid (992) while driving: braked %d ms later", ms);
  rig.run(40);
  check(rig.is("armed", "0") && rig.is("block", "ARM_SWITCH_INVALID") && rig.is("stop", "ARM_SWITCH_INVALID"), "status: armed=0 block=ARM_SWITCH_INVALID");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.is("block", "REARM_REQUIRED") && rig.is("fault", "ARM_SWITCH_INVALID"),
        "ARM back to on from mid: braked, block=REARM_REQUIRED (needs off first)");
  rig.throttle = 992; rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON; rig.run(60);
  rig.throttle = 1811; rig.run(40);
  check(motors().left == 255, "re-armed, driving");

  rig.runToFrame();
  rig.modeRaw = 992;
  ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "MODE channel at mid while driving: braked %d ms later", ms);
  rig.run(40);
  check(rig.is("armed", "1") && rig.is("req", "NONE") && rig.is("act", "NONE") && rig.is("block", "MODE_SWITCH_INVALID") && rig.is("stop", "MODE_SWITCH_INVALID"),
        "status: armed=1 req=NONE act=NONE block=MODE_SWITCH_INVALID");
  rig.pc = PC_FOLLOW; rig.pcThrottle = 0; rig.run(100); rig.pcThrottle = 100;
  check(rig.stoppedFor(500), "PC zero then forward while MODE is invalid: braked");
  rig.pc = PC_SILENT;
  rig.modeRaw = MODE_RADIO;
  check(rig.stoppedFor(500) && rig.is("block", "WAIT_RADIO_NEUTRAL"), "MODE back to RADIO with the stick held: braked, block=WAIT_RADIO_NEUTRAL");

  // Every raw value on each switch channel, from a clean armed state each time.
  int armedOutsideBand = 0, drivenOutsideBand = 0, armedInsideBand = 0;
  for (int raw = 0; raw <= 2047; raw += 7) {
    rig.throttle = 992; rig.modeRaw = MODE_RADIO; rig.armRaw = ARM_OFF; rig.run(12);
    rig.armRaw = raw; rig.run(12);
    rig.throttle = 1811; rig.run(12);
    const bool inBand = band(raw) == (ARM_WHEN_HIGH ? BAND_HIGH : BAND_LOW);
    if (rig.is("armed", "1") && !inBand) ++armedOutsideBand;
    if (!motors().stopped() && !inBand) ++drivenOutsideBand;
    if (!motors().stopped() && inBand) ++armedInsideBand;
  }
  check(armedOutsideBand == 0 && drivenOutsideBand == 0 && armedInsideBand > 50,
        "ARM channel swept 0..2047: arms and drives only for 1400..1950 (%d values drove)", armedInsideBand);
  int drivenInvalidMode = 0, radioDrive = 0;
  for (int raw = 0; raw <= 2047; raw += 7) {
    rig.throttle = 992; rig.modeRaw = MODE_RADIO; rig.armRaw = ARM_OFF; rig.run(12);
    rig.armRaw = ARM_ON; rig.modeRaw = raw; rig.run(12);
    rig.throttle = 1811; rig.run(12);
    if (!motors().stopped() && band(raw) != BAND_LOW) ++drivenInvalidMode;   // BAND_HIGH = PC: no PC client, so no drive
    if (!motors().stopped() && band(raw) == BAND_LOW) ++radioDrive;
  }
  check(drivenInvalidMode == 0 && radioDrive > 50, "MODE channel swept 0..2047: the stick drives only for 100..600 (%d values)", radioDrive);
}

static void pc_cannot_arm_or_select() {
  rig.boot();
  rig.radioOn = true; rig.armRaw = ARM_OFF; rig.modeRaw = MODE_RADIO;
  rig.run(200);
  int n = 0;
  rig.pc = PC_SCRIPT;
  rig.script = [&n](Rig& r) {
    char text[80];
    const unsigned long tok = r.token();
    switch (n++ % 8) {
      case 0: return std::string("ARM\n");
      case 1: return std::string("MODE,PC\n");
      case 2: std::snprintf(text, sizeof(text), "D,%lu,0,0\n", tok); return std::string(text);
      case 3: std::snprintf(text, sizeof(text), "D,%lu,100,0\n", tok); return std::string(text);
      case 4: return std::string("A,1\n");
      case 5: return std::string("ST armed=1 req=PC\n");
      case 6: std::snprintf(text, sizeof(text), "D,%lu,0,0\n", tok + 1); return std::string(text);
      default: return std::string("?\n");
    }
  };
  check(rig.stoppedFor(2000), "disarmed, RADIO selected, PC sends everything it can for 2 s: braked");
  check(rig.is("armed", "0") && rig.is("req", "RADIO") && rig.is("block", "DISARMED"), "status: armed=0 req=RADIO block=DISARMED");
  rig.armRaw = ARM_ON; rig.run(60);
  check(rig.is("armed", "1") && rig.is("act", "RADIO"), "radio arms in RADIO");
  check(rig.stoppedFor(1000) && rig.is("req", "RADIO") && rig.is("pc_ok", "0"), "armed in RADIO, PC still sending: braked (sticks neutral), req=RADIO pc_ok=0");
  rig.throttle = 1500; rig.run(40);
  int l, r;
  expectMix(expectNorm(1500), 0, l, r);
  check(motors().left == l && motors().right == r, "stick forward: L=R=%d from the stick, not the PC's 100", l);
  rig.armRaw = ARM_OFF; rig.modeRaw = MODE_PC; rig.throttle = 992; rig.run(100);
  check(rig.stoppedFor(1000) && rig.is("armed", "0") && rig.is("req", "PC"), "disarmed with PC selected, PC sending zero and forward with the live token: braked, armed=0");
}

static void pc_limit() {
  rig.boot();
  check(rig.armInPc(), "armed in PC");
  rig.pcThrottle = 255; rig.run(60);
  check(motors().left == PC_COMMAND_LIMIT && motors().right == PC_COMMAND_LIMIT, "PC asks for 255: firmware clamps to %d", PC_COMMAND_LIMIT);
  rig.pcThrottle = -255; rig.run(60);
  check(motors().left == -PC_COMMAND_LIMIT && motors().right == -PC_COMMAND_LIMIT, "PC asks for -255: clamped to %d", -PC_COMMAND_LIMIT);
  rig.pcThrottle = 255; rig.pcSteer = 255; rig.run(60);
  int l, r;
  expectMix(PC_COMMAND_LIMIT, PC_COMMAND_LIMIT, l, r);
  check(motors().left == l && motors().right == r && motors().rightBrake, "throttle and steer both 255: L=%d R=%d (clamped, then mixed like the sticks)", l, r);
}

static void usb_not_reading() {
  rig.boot();
  HWCDCSerial.tx_space = 0;                 // nothing connected, or a PC that never reads
  rig.radioOn = true; rig.armRaw = ARM_OFF; rig.run(120);
  rig.armRaw = ARM_ON; rig.run(60);
  rig.throttle = 1811; rig.run(40);
  check(motors().left == 255 && motors().right == 255, "USB output blocked: RADIO still arms and drives");
  check(rig.capture.find('\n', rig.capture.find("HELLO")) == rig.capture.rfind('\n'), "... and nothing was written to USB after boot (never waits for the PC)");
  rig.runToFrame();
  rig.radioOn = false;
  check(rig.msUntilStopped(400) == 251, "... and radio loss still brakes at 251 ms");
}

static void diagnostics_rate() {
  rig.boot();
  rig.radioOn = true; rig.armRaw = ARM_OFF; rig.run(1000);
  int st0 = rig.stLines, ch0 = rig.chLines;
  rig.run(10000);
  check(rig.stLines - st0 >= 48 && rig.stLines - st0 <= 52, "disarmed and steady for 10 s: %d ST lines (5 per second)", rig.stLines - st0);
  check(rig.chLines - ch0 >= 38 && rig.chLines - ch0 <= 41, "... and %d CH lines (4 per second while disarmed)", rig.chLines - ch0);
  std::printf("  example: ST %s...\n", rig.capture.substr(rig.capture.rfind("ST ") + 3, 150).c_str());
  std::printf("  example: %s", rig.capture.substr(rig.capture.rfind("CH "), 240).c_str());
  check(rig.ch["arm"] == "CH5:191:DISARM" && rig.ch["mode"] == "CH6:191:RADIO" && rig.ch["ch5"] == "191" && rig.ch["ch6"] == "191",
        "CH line shows raw switch values and how they decode: arm=%s mode=%s", rig.ch["arm"].c_str(), rig.ch["mode"].c_str());
  rig.armRaw = 1792; rig.modeRaw = 992; rig.run(1200);
  check(rig.ch["arm"] == "CH5:1792:ARM" && rig.ch["mode"] == "CH6:992:INVALID", "after moving them: arm=%s mode=%s", rig.ch["arm"].c_str(), rig.ch["mode"].c_str());
  check(motors().stopped() && rig.is("block", "MODE_SWITCH_INVALID"), "(armed with MODE at mid: braked, block=MODE_SWITCH_INVALID)");
  rig.armRaw = ARM_OFF; rig.modeRaw = MODE_RADIO; rig.run(200);
  rig.armRaw = ARM_ON; rig.run(1000);
  st0 = rig.stLines; ch0 = rig.chLines;
  rig.run(10000);
  check(rig.chLines - ch0 >= 9 && rig.chLines - ch0 <= 11, "armed for 10 s: %d CH lines (1 per second)", rig.chLines - ch0);
  st0 = rig.stLines;
  for (int i = 0; i < 1250; ++i) {          // MODE flipped every radio frame for 5 s
    rig.modeRaw = (i % 2) ? MODE_PC : MODE_RADIO;
    rig.run(4);
  }
  check(rig.stLines - st0 <= 251, "MODE flapping every 4 ms for 5 s: %d ST lines (at most 50 per second)", rig.stLines - st0);
  check(rig.badLines == 0 && rig.longest < 250, "every USB line was a well-formed ST, CH or HELLO line; longest %d bytes", rig.longest);
}

// ---------------------------------------------------------------- Wi-Fi scenarios

static bool heldAt(int left, int right, int ms) {
  bool all = true;
  for (int i = 0; i < ms; ++i) {
    rig.tick();
    const Motors m = motors();
    all = all && m.valid && m.left == left && m.right == right;
  }
  return all;
}

static void wifi_drive_without_usb() {
  HWCDCSerial.tx_space = 0;                 // USB unplugged from power-up: nothing can ever be written to it
  rig.boot();
  check(mock::ap_started && mock::ap_ssid == WIFI_AP_SSID && mock::ap_password.size() >= 8 && mock::ap_channel == WIFI_AP_CHANNEL &&
        mock::ap_hidden == 0 && mock::ap_max_connections == 1,
        "access point started: \"%s\", password-protected, channel %d, one laptop at most", mock::ap_ssid.c_str(), mock::ap_channel);
  check(mock::udp_port == WIFI_UDP_PORT && !mock::modem_sleep && mock::tx_power == (WIFI_REDUCED_TX_POWER ? WIFI_POWER_8_5dBm : -1), "UDP port %d open, Wi-Fi power-save off, transmit power as configured", mock::udp_port);
  check(ELRS.rx_buffer_size == 1024 && ELRS.begun_baud == 420000, "radio UART buffer enlarged before begin()");
  check(rig.stoppedFor(500) && rig.wifiOutCount == 0, "no laptop has spoken: braked, and the robot transmits nothing to anyone");

  rig.radioOn = true; rig.armRaw = ARM_OFF; rig.modeRaw = MODE_PC;
  rig.wifi = Rig::WIFI_FOLLOW;
  rig.run(700);
  check(rig.wifiHelloLines >= 1 && rig.wifiHello.find("fw=autobot-handoff-wifi-1") != std::string::npos && rig.wifiHello.find("wifi_ok=1") != std::string::npos,
        "\"?\" over Wi-Fi gets the HELLO line: %s", rig.wifiHello.c_str());
  rig.wifiStatus("laptop joined, disarmed");
  check(rig.wis("link", "1") && rig.wis("armed", "0") && rig.wis("block", "DISARMED") && rig.wis("req", "PC") && rig.wis("act", "NONE"),
        "status arrives over Wi-Fi with USB dead: link=1 armed=0 req=PC act=NONE block=DISARMED");
  check(rig.wn("ap") == "1" && rig.wn("sta") == "1" && rig.wn("peer") == "192.168.4.2:50123" && rig.wn("pc_via") == "NONE",
        "NET line: ap=1 sta=1 peer=192.168.4.2:50123 pc_via=NONE");
  rig.wifiThrottle = 100;
  check(rig.stoppedFor(1000) && rig.wis("armed", "0"), "laptop sends forward while disarmed: braked (the PC cannot arm)");
  rig.wifiThrottle = 0;
  rig.armRaw = ARM_ON;
  rig.run(150);
  check(rig.wis("armed", "1") && rig.wis("act", "PC") && rig.wis("pc_ok", "1") && rig.wn("pc_via") == "WIFI" && motors().stopped(),
        "RadioMaster arms with PC selected, laptop neutral: act=PC pc_ok=1 pc_via=WIFI, braked");
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "forward 100 from the laptop: L=R=100");
  rig.wifiThrottle = -100; rig.run(60);
  check(motors().left == -100 && motors().right == -100, "reverse: L=R=-100");
  rig.wifiThrottle = 0; rig.wifiSteer = 60; rig.run(60);
  check(motors().left == 60 && motors().right == -60, "turn right 60: L=60 R=-60");
  rig.wifiSteer = -60; rig.run(60);
  check(motors().left == -60 && motors().right == 60, "turn left 60: L=-60 R=60");
  rig.wifiSteer = 0; rig.run(60);
  check(motors().stopped() && rig.wis("pc_ok", "1") && rig.wis("act", "PC"), "keys released (zero): braked, the laptop keeps control");
  rig.wifiThrottle = 255; rig.run(60);
  check(motors().left == PC_COMMAND_LIMIT, "laptop asks for 255: clamped to %d, as over USB", PC_COMMAND_LIMIT);
  check(rig.capture.empty() && rig.stLines == 0, "nothing was ever written to USB in this whole run");
}

static void wifi_link_drop_latches() {
  HWCDCSerial.tx_space = 0;
  rig.boot();
  check(rig.armInPcWifi(), "USB dead; armed in PC, Wi-Fi laptop in control");
  rig.throttle = 1811;                      // radio stick held at full throughout: it must never matter
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "laptop driving forward 100");
  rig.runToWifiSend();                      // the last datagram that arrives
  const size_t sentBefore = rig.wifiSent.size();
  rig.wifiReachable = false;                // the laptop drops off the robot's Wi-Fi with the key still held
  check(rig.movingFor(249), "Wi-Fi drops while commanding motion: still driving 249 ms after the last command");
  rig.tick();
  check(motors().stopped(), "... braked at 250 ms, although the last command was non-zero");
  rig.run(40);
  check(!armed && fault == CAUSE_PC_STALE && block == BLOCK_REARM_REQUIRED && lastStop == CAUSE_PC_STALE && pcVia == PC_VIA_NONE,
        "robot state: disarmed, fault PC_STALE, block REARM_REQUIRED, stop PC_STALE, nobody holds the PC input");
  check(rig.stoppedFor(1500), "1.5 s more with the radio stick at full: braked (no fall-back to RADIO)");

  rig.wifiReachable = true;                 // the laptop rejoins; the client is still sending forward
  check(rig.stoppedFor(1000), "laptop back on the network and still sending forward: braked for 1 s (no automatic motion)");
  rig.wifiStatus("after reconnect");
  check(rig.wis("armed", "0") && rig.wis("act", "NONE") && rig.wis("pc_ok", "0") && rig.wis("block", "REARM_REQUIRED") &&
        rig.wis("fault", "PC_STALE") && rig.wis("stop", "PC_STALE") && rig.wn("pc_via") == "NONE",
        "status over Wi-Fi: armed=0 act=NONE pc_ok=0 block=REARM_REQUIRED fault=PC_STALE stop=PC_STALE pc_via=NONE");

  rig.wifi = Rig::WIFI_OFF;
  bool replayStopped = true;
  for (size_t i = 0; i < sentBefore; ++i) { // every datagram from before the drop, byte for byte
    rig.sendWifiRaw(rig.wifiSent[i]);
    if (i % 4 == 3) { rig.tick(); replayStopped = replayStopped && motors().stopped(); }
  }
  check(replayStopped && rig.stoppedFor(300), "all %d datagrams from before the drop replayed: braked", static_cast<int>(sentBefore));

  rig.wifi = Rig::WIFI_FOLLOW; rig.wifiThrottle = 0; rig.run(200);
  rig.wifiThrottle = 100;
  check(rig.stoppedFor(500) && rig.wis("pc_ok", "0") && rig.wis("armed", "0"), "zero then forward with the live token, no re-arm: braked (the PC cannot clear the fault)");

  rig.armRaw = ARM_OFF; rig.run(100);
  check(rig.wis("fault", "NONE") && rig.wis("block", "DISARMED"), "RadioMaster ARM off: fault cleared");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.wis("block", "WAIT_PC_NEUTRAL") && rig.wis("armed", "1"), "ARM on, laptop key still held: braked, block=WAIT_PC_NEUTRAL");
  rig.wifi = Rig::WIFI_OFF;
  for (size_t i = 0; i < sentBefore; ++i) { // the old datagrams again, now that it is armed and waiting for a neutral
    rig.sendWifiRaw(rig.wifiSent[i]);
    if (i % 4 == 3) { rig.tick(); replayStopped = replayStopped && motors().stopped(); }
  }
  rig.run(100);
  check(replayStopped && motors().stopped() && !granted, "old datagrams (they include old zero commands) replayed while waiting: not accepted as the neutral");
  rig.wifi = Rig::WIFI_FOLLOW;
  rig.wifiThrottle = 0; rig.run(60);
  check(rig.wis("pc_ok", "1") && motors().stopped(), "fresh zero with the live token: pc_ok=1");
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "then forward: drives again");
}

static void wifi_radio_takeover_while_sending() {
  HWCDCSerial.tx_space = 0;
  rig.boot();
  check(rig.armInPcWifi(), "USB dead; armed in PC, Wi-Fi laptop in control");
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "laptop driving forward 100");

  rig.runToFrame();
  rig.modeRaw = MODE_RADIO;                 // takeover; the laptop never stops sending forward
  int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "MODE to RADIO while the laptop keeps sending: braked %d ms later", ms);
  rig.run(60);
  rig.wifiStatus("after takeover");
  check(rig.wis("req", "RADIO") && rig.wis("act", "RADIO") && rig.wis("pc_ok", "0") && rig.wis("stop", "HANDOFF") && rig.wn("pc_via") == "NONE" && motors().stopped(),
        "sticks were centred: act=RADIO pc_ok=0 stop=HANDOFF pc_via=NONE, braked");
  rig.throttle = 1500;
  int l, r;
  expectMix(expectNorm(1500), 0, l, r);
  rig.run(20);
  check(heldAt(l, r, 2000), "stick forward for 2 s: L=R=%d from the stick on every millisecond, laptop's forward 100 ignored", l);
  check(std::atoi(rig.ws("pc_age").c_str()) >= 0 && std::atoi(rig.ws("pc_age").c_str()) <= 25, "pc_age=%s ms shows the laptop was still sending all along", rig.ws("pc_age").c_str());

  rig.throttle = 992; rig.run(40);
  rig.modeRaw = MODE_PC;                    // back to PC with the laptop key still held
  check(rig.stoppedFor(1000) && rig.wis("block", "WAIT_PC_NEUTRAL"), "MODE back to PC, laptop key held: braked for 1 s, block=WAIT_PC_NEUTRAL");
  rig.wifiThrottle = 0; rig.run(60);
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100 && rig.wn("pc_via") == "WIFI", "laptop neutral, then forward: drives, pc_via=WIFI");

  rig.throttle = 1811;                      // takeover again, this time with the radio stick held
  rig.runToFrame();
  rig.modeRaw = MODE_RADIO;
  ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4 && rig.stoppedFor(1000) && rig.wis("block", "WAIT_RADIO_NEUTRAL"),
        "takeover with the stick already at full: braked, block=WAIT_RADIO_NEUTRAL until the stick is centred");
  rig.throttle = 992; rig.run(20);
  rig.throttle = 1811; rig.run(20);
  check(motors().left == 255 && motors().right == 255, "stick centred, then forward: the RadioMaster drives");
}

static void wifi_disarm_and_radio_loss() {
  HWCDCSerial.tx_space = 0;
  rig.boot();
  check(rig.armInPcWifi(), "USB dead; armed in PC, Wi-Fi laptop in control");
  rig.wifiThrottle = 100; rig.run(60);
  rig.runToFrame();
  rig.armRaw = ARM_OFF;
  int ms = rig.msUntilStopped(50);
  check(ms >= 0 && ms <= 4, "ARM off while the laptop drives: braked %d ms later", ms);
  check(rig.stoppedFor(1000) && rig.wis("armed", "0") && rig.wis("block", "DISARMED") && rig.wis("stop", "DISARM"),
        "laptop keeps sending forward for 1 s: braked, block=DISARMED stop=DISARM");
  rig.armRaw = ARM_ON;
  check(rig.stoppedFor(500) && rig.wis("block", "WAIT_PC_NEUTRAL"), "ARM on with the key held: braked, block=WAIT_PC_NEUTRAL");
  rig.wifiThrottle = 0; rig.run(60);
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "neutral, then forward: drives");

  rig.runToFrame();
  rig.radioOn = false;                      // transmitter off; the laptop keeps sending forward
  check(rig.movingFor(250), "transmitter off in PC mode: still driving 250 ms after the last radio frame");
  rig.tick();
  check(motors().stopped(), "... braked at 251 ms although Wi-Fi commands keep arriving");
  check(rig.stoppedFor(1000) && rig.wis("link", "0") && rig.wis("block", "RADIO_LOST") && rig.wis("stop", "RADIO_LOST"),
        "1 s more: braked; status still reaches the laptop: link=0 block=RADIO_LOST stop=RADIO_LOST");
  rig.radioOn = true;
  check(rig.stoppedFor(1000) && rig.wis("block", "REARM_REQUIRED") && rig.wis("fault", "RADIO_LOST"),
        "transmitter back, laptop still sending: braked, block=REARM_REQUIRED fault=RADIO_LOST");
}

static void wifi_and_usb_never_command_together() {
  rig.boot();
  check(rig.armInPc(), "armed in PC; the USB client supplied the neutral");
  rig.run(600);
  check(rig.un("pc_via") == "USB", "NET line: pc_via=USB");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100 && motors().right == 100, "USB client driving forward 100");
  rig.wifi = Rig::WIFI_FOLLOW; rig.wifiThrottle = -100;      // a Wi-Fi laptop appears and sends reverse with the live token
  check(heldAt(100, 100, 1000), "Wi-Fi laptop sends reverse with the live token for 1 s: L=R=100 on every millisecond");
  check(pcOtherTransport > 20, "those Wi-Fi commands were counted and ignored (%lu)", static_cast<unsigned long>(pcOtherTransport));
  rig.wifiThrottle = 0; rig.run(100);
  rig.wifiThrottle = -100;
  check(heldAt(100, 100, 500), "Wi-Fi zero, then reverse: still L=R=100 (a neutral from the other transport takes nothing over)");

  rig.runToPcSend();
  rig.pc = PC_SILENT;                        // the USB client dies; the Wi-Fi laptop keeps sending valid commands
  check(heldAt(100, 100, 249), "USB client stops: still its last command for 249 ms");
  rig.tick();
  check(motors().stopped(), "... braked at 250 ms; the Wi-Fi commands did not keep the PC link alive");
  check(rig.stoppedFor(1000) && rig.is("fault", "PC_STALE") && rig.is("block", "REARM_REQUIRED"),
        "1 s more of Wi-Fi reverse: braked, fault=PC_STALE (no silent switch to the other transport)");

  rig.wifiThrottle = 0;                      // now the other way round: Wi-Fi neutral, USB key held
  rig.pc = PC_FOLLOW; rig.pcThrottle = 100;
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON; rig.run(700);
  check(rig.is("pc_ok", "1") && rig.un("pc_via") == "WIFI" && motors().stopped(), "re-armed: only the Wi-Fi laptop was neutral, so it holds the PC input (pc_via=WIFI)");
  rig.wifiThrottle = 60; rig.run(60);
  check(heldAt(60, 60, 1000), "Wi-Fi forward 60 while USB keeps sending 100: L=R=60 on every millisecond");
  rig.runToWifiSend();
  rig.wifi = Rig::WIFI_OFF;
  check(heldAt(60, 60, 249), "Wi-Fi laptop stops: still its last command for 249 ms");
  rig.tick();
  check(motors().stopped() && rig.stoppedFor(1000), "... braked at 250 ms and for 1 s more, with USB still sending 100");
}

static void wifi_old_duplicate_and_foreign_datagrams() {
  rig.boot();
  check(rig.armInPcWifi(), "armed in PC, Wi-Fi laptop in control");
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "laptop driving forward 100");
  rig.runToWifiSend();
  rig.wifi = Rig::WIFI_OFF;                  // from here the datagrams are sent by hand
  const unsigned long tok = rig.wifiToken();
  char text[120];

  std::snprintf(text, sizeof(text), "D,%lu,-100,0,%lu\n", tok, static_cast<unsigned long>(rig.wifiSeq - 3));
  rig.sendWifiRaw(text); rig.run(5);
  check(motors().left == 100 && wifiOld == 1, "late datagram (live token, older sequence number, reverse): ignored");
  rig.sendWifiRaw(rig.wifiSent.back()); rig.run(5);
  check(motors().left == 100 && wifiOld == 2, "exact duplicate of the newest datagram: ignored");
  rig.wifiSeq += 5;
  rig.sendWifiDrive(tok, 50, 0); rig.run(5);
  std::snprintf(text, sizeof(text), "D,%lu,-100,0,%lu\n", tok, static_cast<unsigned long>(rig.wifiSeq - 2));
  rig.sendWifiRaw(text); rig.run(5);
  check(motors().left == 50 && wifiOld == 3, "out of order (a skipped sequence number arriving late): ignored, newest command 50 stands");

  std::snprintf(text, sizeof(text), "D,%lu,-100,0,%lu\n", tok, static_cast<unsigned long>(rig.wifiSeq + 100));
  rig.sendWifiDrive(tok, 50, 0); rig.tick();          // the owner's last command
  int stoppedAfter = -1;
  for (int i = 1; i <= 400 && stoppedAfter < 0; ++i) {
    if (i % 20 == 0) rig.sendWifiRaw(text, IPAddress(192, 168, 4, 3), 6000);   // another address, live token, newer sequence
    rig.tick();
    const Motors m = motors();
    if (m.stopped()) stoppedAfter = i;
    else if (m.left != 50) stoppedAfter = -100;
  }
  check(stoppedAfter == 250, "another address sends reverse with the live token: never applied, and it does not keep the link alive (braked at %d ms)", stoppedAfter);
  check(pcOtherTransport >= 10 && rig.wifiStray == 0, "its datagrams were counted and ignored, and no status line was sent to it");

  rig.wifi = Rig::WIFI_FOLLOW; rig.wifiThrottle = 0;
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON; rig.run(150);
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "re-armed, driving 100 again");
  rig.runToWifiSend();
  rig.wifi = Rig::WIFI_OFF;
  const unsigned long live = rig.wifiToken();
  const uint32_t badBefore = wifiBad;
  int sent = 0;
  stoppedAfter = -1;
  for (int i = 1; i <= 400 && stoppedAfter < 0; ++i) {
    if (i % 20 == 0) {
      const unsigned long seq = rig.wifiSeq + 1000 + i;
      switch (sent++ % 10) {
        case 0: std::snprintf(text, sizeof(text), "D,%lu,100,0\n", live); break;                    // USB format: no sequence
        case 1: std::snprintf(text, sizeof(text), "D,%lu,100,0,\n", live); break;
        case 2: std::snprintf(text, sizeof(text), "D,%lu,100,0,%lu,7\n", live, seq); break;         // extra field
        case 3: std::snprintf(text, sizeof(text), "D,%lu,100,0,x\n", live); break;
        case 4: std::snprintf(text, sizeof(text), "D,%lu,300,0,%lu\n", live, seq); break;           // out of range
        case 5: std::snprintf(text, sizeof(text), "D,%lu,100,0,%lu\nD,%lu,100,0,%lu\n", live, seq, live, seq + 1); break;   // two lines
        case 6: std::snprintf(text, sizeof(text), "D,%lu,100,0,%lu                                          \n", live, seq); break;  // over-long
        case 7: std::snprintf(text, sizeof(text), "\x01\x02\x03"); break;
        case 8: std::snprintf(text, sizeof(text), " D,%lu,100,0,%lu\n", live, seq); break;
        default: std::snprintf(text, sizeof(text), "d,%lu,100,0,%lu\n", live, seq); break;
      }
      rig.sendWifiRaw(text);
    }
    rig.tick();
    if (motors().stopped()) stoppedAfter = i;
  }
  check(stoppedAfter == 250, "only malformed datagrams from here: braked %d ms after the last good command", stoppedAfter);
  check(wifiBad - badBefore == static_cast<uint32_t>(sent), "all %d malformed datagrams were counted as bad (%lu)", sent, static_cast<unsigned long>(wifiBad - badBefore));

  rig.wifi = Rig::WIFI_FOLLOW; rig.wifiThrottle = 0;
  rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON; rig.run(150);
  rig.wifiThrottle = 100; rig.run(60);
  check(motors().left == 100, "after the over-long and split datagrams the socket still works: re-armed and driving");
  const uint32_t usbBad = pcBadLines;
  std::snprintf(text, sizeof(text), "D,%lu,-100,0,%lu\n", rig.wifiToken(), static_cast<unsigned long>(rig.wifiSeq + 50));
  rig.sendPc(text); rig.run(10);
  check(pcBadLines == usbBad + 1 && motors().left == 100, "the Wi-Fi format sent over USB is rejected there (USB keeps its tested 4-field format)");
}

static void wifi_flood_cannot_delay_the_radio() {
  rig.boot();
  check(rig.armInRadio(), "armed in RADIO");
  rig.throttle = 1811; rig.run(40);
  check(motors().left == 255, "RadioMaster driving full forward");
  uint32_t mostPerMs = 0;
  auto floodTick = [&mostPerMs] {
    for (int i = 0; i < 200; ++i) mock::wifi_in.push_back(mock::Datagram{IPAddress(192, 168, 4, 2), 50123, "D,1,100,0,1\n"});
    const uint32_t before = wifiReceived;
    rig.tick();
    mostPerMs = std::max(mostPerMs, wifiReceived - before);
  };
  for (int i = 0; i < 200; ++i) floodTick();
  check(motors().left == 255 && rig.violations == 0, "200 datagrams a millisecond for 0.2 s: radio drive unaffected");
  rig.runToFrame();
  rig.radioOn = false;
  int stoppedAfter = -1;
  for (int i = 1; i <= 400 && stoppedAfter < 0; ++i) {
    floodTick();
    if (motors().stopped()) stoppedAfter = i;
  }
  check(stoppedAfter == 251, "transmitter off during the flood: braked at %d ms, exactly as without Wi-Fi traffic", stoppedAfter);
  check(mostPerMs <= WIFI_MAX_DATAGRAMS && mock::wifi_in.size() > 50000, "at most %lu datagrams were handled per millisecond (limit %d), %d left waiting",
        static_cast<unsigned long>(mostPerMs), WIFI_MAX_DATAGRAMS, static_cast<int>(mock::wifi_in.size()));
  mock::wifi_in.clear();
  rig.radioOn = true; rig.throttle = 992; rig.armRaw = ARM_OFF; rig.run(100);
  rig.armRaw = ARM_ON; rig.run(60);
  rig.throttle = 1811; rig.run(40);
  rig.runToFrame();
  rig.armRaw = ARM_OFF;
  stoppedAfter = -1;
  for (int i = 0; i <= 50 && stoppedAfter < 0; ++i) {
    if (motors().stopped()) stoppedAfter = i;
    else floodTick();
  }
  check(stoppedAfter >= 0 && stoppedAfter <= 4, "ARM off during a flood: braked %d ms later", stoppedAfter);
}

static void wifi_access_point_failure_changes_nothing_else() {
  mock::ap_fails = true;                    // the access point does not start
  rig.boot();
  rig.sendWifiRaw("?\n");
  rig.run(700);
  check(rig.hello.find("wifi_ok=0") != std::string::npos && rig.un("ap") == "0" && rig.wifiOutCount == 0,
        "HELLO says wifi_ok=0, NET says ap=0, nothing is read from or sent to the network");
  check(rig.armInRadio(), "RadioMaster still arms");
  rig.throttle = 1811; rig.run(20);
  check(motors().left == 255, "... and drives");
  rig.throttle = 992;
  check(rig.armInPc(), "USB client still gets PC control");
  rig.pcThrottle = 100; rig.run(60);
  check(motors().left == 100 && rig.un("pc_via") == "USB", "... and drives over USB");
}

static void wifi_status_follows_the_laptop() {
  rig.boot();
  rig.radioOn = true;
  rig.run(1000);
  check(rig.wifiOutCount == 0, "no laptop has spoken for 1 s: nothing transmitted over Wi-Fi");
  rig.wifi = Rig::WIFI_FOLLOW;
  rig.run(700);
  check(rig.wifiHelloLines >= 1 && rig.wst.count("tok") == 1, "first \"?\": HELLO and status start arriving");
  const int st0 = rig.wifiStLines, net0 = rig.wifiNetLines, ch0 = rig.wifiChLines, usb0 = rig.stLines;
  rig.run(4000);
  check(rig.wifiStLines - st0 >= 19 && rig.wifiNetLines - net0 >= 7 && rig.wifiChLines - ch0 >= 14,
        "4 s: %d ST, %d NET, %d CH lines over Wi-Fi", rig.wifiStLines - st0, rig.wifiNetLines - net0, rig.wifiChLines - ch0);
  check(std::abs((rig.stLines - usb0) - (rig.wifiStLines - st0)) <= 1 && rig.wifiBadLines == 0 && rig.wifiStray == 0,
        "the same ST lines went to USB (%d) and Wi-Fi (%d), all well-formed, none to any other address", rig.stLines - usb0, rig.wifiStLines - st0);
  check(rig.token() == rig.wifiToken(), "both transports report the same token");
  rig.wifi = Rig::WIFI_OFF;                 // laptop goes quiet but stays on the network
  rig.stationsForced = 1;
  rig.run(3700);
  const int out = rig.wifiOutCount;
  rig.run(1000);
  check(rig.wifiOutCount == out && rig.un("peer") == "NONE" && rig.un("sta") == "1",
        "3 s without a datagram from it: status stops going out; USB NET line says peer=NONE sta=1");
}

// ---------------------------------------------------------------- cosim
//
// stdin, one line per step:  <ms> <radio 0|1> <ch1> <ch2> <arm raw> <mode raw> <usb hex or -> <wifi datagram hex or ->
// stdout, one line per step: <L> <R> <stopped 0|1> <violations> <usb hex or -> <wifi hex or ->
// A Wi-Fi datagram is delivered from 192.168.4.2:50123; "-" means nothing arrived (laptop silent or off the network).

static std::string unhex(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  return out;
}

static std::string tohex(const std::string& text, size_t from) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (size_t i = from; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    out.push_back(digits[c >> 4]);
    out.push_back(digits[c & 15]);
  }
  return out.empty() ? "-" : out;
}

static void feedOracle(const std::string& line, bool wifiLine) {
  unsigned long tok = 0, seq = 0;
  int t = 0, s = 0;
  const int got = std::sscanf(line.c_str(), "D,%lu,%d,%d,%lu", &tok, &t, &s, &seq);
  if (got < 3) return;
  if (wifiLine) { rig.prevWifiT = rig.lastWifiT; rig.prevWifiS = rig.lastWifiS; rig.lastWifiT = t; rig.lastWifiS = s; }
  else { rig.prevPcT = rig.lastPcT; rig.prevPcS = rig.lastPcS; rig.lastPcT = t; rig.lastPcS = s; }
  rig.lastPcSentMs = mock::now_ms;
  rig.pcSentEver = true;
  if (t == 0 && s == 0) rig.zeroPcSinceChange = true;
}

static int cosim() {
  rig.boot();
  rig.stationsForced = 0;
  unsigned long lastWifi = 0;
  bool wifiEver = false;
  std::string text;
  while (std::getline(std::cin, text)) {
    std::istringstream in(text);
    int ms = 0, on = 0;
    std::string usbHex, wifiHex;
    if (!(in >> ms >> on >> rig.steer >> rig.throttle >> rig.armRaw >> rig.modeRaw >> usbHex >> wifiHex)) break;
    rig.radioOn = on != 0;
    if (usbHex != "-") {
      const std::string decoded = unhex(usbHex);
      for (char c : decoded) HWCDCSerial.rx.push_back(static_cast<uint8_t>(c));
      std::istringstream lines(decoded);
      std::string one;
      while (std::getline(lines, one)) feedOracle(one, false);
    }
    if (wifiHex != "-") {
      const std::string decoded = unhex(wifiHex);
      rig.sendWifiRaw(decoded);
      feedOracle(decoded, true);
      lastWifi = mock::now_ms;
      wifiEver = true;
    }
    rig.stationsForced = (wifiEver && mock::now_ms - lastWifi < 1000) ? 1 : 0;
    const size_t usbFrom = rig.capture.size(), wifiFrom = rig.wifiCapture.size();
    rig.run(ms);
    const Motors m = motors();
    std::printf("%d %d %d %d %s %s\n", m.left, m.right, m.stopped() ? 1 : 0, rig.violations,
                tohex(rig.capture, usbFrom).c_str(), tohex(rig.wifiCapture, wifiFrom).c_str());
    std::fflush(stdout);
  }
  return rig.violations == 0 && rig.badLines == 0 && rig.wifiBadLines == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- main

struct Scenario {
  const char* name;
  void (*run)();
};

static const Scenario scenarios[] = {
  {"boot_no_radio", boot_no_radio},
  {"boot_arm_already_on", boot_arm_already_on},
  {"boot_arm_already_on_pc", boot_arm_already_on_pc},
  {"drive_matches_original", drive_matches_original},
  {"stop_path", stop_path},
  {"original_stop_path_model", original_stop_path_model},
  {"disarm_radio", disarm_radio},
  {"disarm_pc", disarm_pc},
  {"handoff_radio_to_pc", handoff_radio_to_pc},
  {"handoff_pc_to_radio", handoff_pc_to_radio},
  {"handoff_flapping", handoff_flapping},
  {"pc_stale", pc_stale},
  {"pc_garbage_is_not_alive", pc_garbage_is_not_alive},
  {"pc_reconnect_old_command", pc_reconnect_old_command},
  {"radio_loss_in_radio", radio_loss_in_radio},
  {"radio_loss_in_pc", radio_loss_in_pc},
  {"radio_reports_zero_link_quality", radio_reports_zero_link_quality},
  {"radio_noise_is_not_a_link", radio_noise_is_not_a_link},
  {"short_rc_frame", short_rc_frame},
  {"switch_invalid", switch_invalid},
  {"pc_cannot_arm_or_select", pc_cannot_arm_or_select},
  {"pc_limit", pc_limit},
  {"usb_not_reading", usb_not_reading},
  {"diagnostics_rate", diagnostics_rate},
  {"wifi_drive_without_usb", wifi_drive_without_usb},
  {"wifi_link_drop_latches", wifi_link_drop_latches},
  {"wifi_radio_takeover_while_sending", wifi_radio_takeover_while_sending},
  {"wifi_disarm_and_radio_loss", wifi_disarm_and_radio_loss},
  {"wifi_and_usb_never_command_together", wifi_and_usb_never_command_together},
  {"wifi_old_duplicate_and_foreign_datagrams", wifi_old_duplicate_and_foreign_datagrams},
  {"wifi_flood_cannot_delay_the_radio", wifi_flood_cannot_delay_the_radio},
  {"wifi_access_point_failure_changes_nothing_else", wifi_access_point_failure_changes_nothing_else},
  {"wifi_status_follows_the_laptop", wifi_status_follows_the_laptop},
};

int main(int argc, char** argv) {
  const std::string want = argc > 1 ? argv[1] : "--list";
  if (want == "--list") {
    for (const Scenario& s : scenarios) std::printf("%s\n", s.name);
    return 0;
  }
  if (want == "cosim") return cosim();
  for (const Scenario& s : scenarios) {
    if (want != s.name) continue;
    std::printf("%s\n", s.name);
    s.run();
    // The original sketch's own scenarios drive the pins outside the rig's rules.
    const bool usesRig = want != "original_stop_path_model";
    if (usesRig) {
      check(rig.violations == 0, "invariants held on every simulated millisecond (%d broken)", rig.violations);
      check(rig.badLines == 0, "every USB line was well-formed (%d were not)", rig.badLines);
      check(rig.wifiBadLines == 0, "every Wi-Fi line was well-formed (%d were not)", rig.wifiBadLines);
    }
    std::printf("%s: %d checks, %d failed\n", s.name, checks, failures);
    return failures == 0 ? 0 : 1;
  }
  std::printf("unknown scenario: %s\n", want.c_str());
  return 2;
}
