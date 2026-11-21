"""Regenerate ELRS-Handoff-Telemetry/ELRS-Handoff-Telemetry.ino from the Wi-Fi handoff sketch.

The telemetry sketch is the Wi-Fi handoff sketch plus an LSM6DS3 reader and two status lines. It is
produced by exact-text insertions so that the difference from the handoff-only build stays small and
reviewable, and so that your Wi-Fi name, password and channel carry over when you change them:

    python scripts/make_telemetry_sketch.py

Nothing in the handoff, arming, radio or motor code is touched: every edit below is an insertion, except
the firmware name in the HELLO line. tests/test_handoff_telemetry.py checks that.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ELRS-Handoff-WiFi" / "ELRS-Handoff-WiFi.ino"
TARGET = ROOT / "ELRS-Handoff-Telemetry" / "ELRS-Handoff-Telemetry.ino"

CONFIG = r'''
// ---------------------
// IMU TELEMETRY (LSM6DS3 on I2C)
// ---------------------
//
// Read-only: the IMU is sampled and sent to the PC. Nothing in this
// sketch uses it to decide anything about driving.
//
// SDA 6 / SCL 7, 100 kHz, address 0x6A or 0x6B and the three register
// settings are those of the standalone IMU test sketch that found the
// sensor on this robot (17 Nov, "I2C address 0x6B, WHO_AM_I 0x69").
// If the sensor does not answer, the IMU line says so and everything
// else runs as before.

#define IMU_SDA_PIN  6
#define IMU_SCL_PIN  7
#define IMU_I2C_HZ   100000

// The longest one I2C transaction may hold the loop, even with a
// stuck bus. Only one transaction is ever made per loop pass, after
// the motors have been set.
#define IMU_I2C_TIMEOUT_MS  5

#define IMU_PERIOD_MS   20     // 50 samples a second
#define IMU_RETRY_MS    250    // absent or failed: one short probe this often
#define IMU_FAILS_BEFORE_LOST  3

// LSM6DS3 scale factors for the ranges set below (+/-4 g, +/-2000 deg/s).
#define IMU_ACCEL_MS2_PER_COUNT  (0.000122 * 9.80665)
#define IMU_GYRO_DPS_PER_COUNT   0.070
'''

ASSERTS = r'''#define IMU_PIN_IS_FREE(p) \
  ((p) != AIN1 && (p) != AIN2 && (p) != BIN1 && (p) != BIN2 && \
   (p) != ELRS_RX_PIN && (p) != 18 && (p) != 19 && (p) != 8 && (p) != 9)
static_assert(IMU_PIN_IS_FREE(IMU_SDA_PIN), "IMU_SDA_PIN is a motor, radio, USB or strapping pin");
static_assert(IMU_PIN_IS_FREE(IMU_SCL_PIN), "IMU_SCL_PIN is a motor, radio, USB or strapping pin");
static_assert(IMU_SDA_PIN != IMU_SCL_PIN, "IMU_SDA_PIN and IMU_SCL_PIN are the same pin");
'''

STATE = r'''
// ---- IMU telemetry ----

#define IMU_NOT_FOUND      0   // nothing has answered WHO_AM_I
#define IMU_CONFIGURING    1   // found; writing its three settings, one per loop pass
#define IMU_RUNNING        2
#define IMU_READ_FAILED    3   // it was running and stopped answering

uint8_t imuState = IMU_NOT_FOUND;
uint8_t imuAddress = 0;
uint8_t imuWhoAmI = 0;
uint8_t imuConfigStep = 0;
uint8_t imuProbeNext = 0;           // which of the two addresses to try next
uint8_t imuConsecutiveFails = 0;
bool imuBusOk = false;

uint32_t imuSequence = 0;           // counts every sample attempt, valid or not
uint32_t imuGoodReads = 0;
uint32_t imuFailedReads = 0;
unsigned long imuLongestReadUs = 0;

unsigned long lastImuSample = 0;
unsigned long lastImuAttempt = 0;
unsigned long lastImuInfo = 0;
'''

FUNCTIONS = r'''// =====================================================
// IMU TELEMETRY
// =====================================================
//
//   IMU seq=<n> t_us=<microseconds since boot> ok=1 ax=.. ay=.. az=.. gx=.. gy=.. gz=.. unit=m/s2,deg/s
//   IMU seq=<n> t_us=<..> ok=0 why=NOT_FOUND | CONFIG_FAILED | READ_FAILED
//   IMUINFO state=.. addr=.. who=.. hz=.. range=4g,2000dps unit=m/s2,deg/s axes=sensor good=.. failed=.. longest_read_us=..
//
// Values are in the SENSOR's own axes. Nothing here knows how the
// sensor is mounted; the PC applies the mounting transform.
// A sample that could not be read is sent with ok=0 and no numbers.
// It is never sent as zeros.

bool imuReadRegisters(uint8_t address, uint8_t reg, uint8_t *data, uint8_t count) {

  // One combined write-then-read transaction, bounded by Wire's timeout.

  Wire.beginTransmission(address);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(address, count) != count) {
    return false;
  }

  for (uint8_t i = 0; i < count; i++) {

    int value = Wire.read();

    if (value < 0) {
      return false;
    }

    data[i] = (uint8_t)value;
  }

  return true;
}


bool imuWriteRegister(uint8_t address, uint8_t reg, uint8_t value) {

  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);

  return Wire.endTransmission() == 0;
}


const char *imuStateName() {

  switch (imuState) {
    case IMU_CONFIGURING: return "CONFIGURING";
    case IMU_RUNNING:     return "OK";
    case IMU_READ_FAILED: return "READ_FAILED";
    default:              return "NOT_FOUND";
  }
}


// IMU lines never crowd the status lines out of the small USB buffer:
// they go to USB only when it is nearly empty. Over Wi-Fi they always go.

void sendImuLine(const char *line, int size) {

  if (size <= 0 || size >= 250) {
    return;
  }

  if (USB_PORT.availableForWrite() >= size + 120) {
    USB_PORT.write((const uint8_t *)line, (size_t)size);
  }

  if (wifiUdpOk && statusPeerKnown) {

    if (wifiUdp.beginPacket(statusPeerIp, statusPeerPort)) {

      wifiUdp.write((const uint8_t *)line, (size_t)size);
      wifiUdp.endPacket();
    }
  }
}


void sendImuInvalid(const char *why) {

  char out[96];

  int size = snprintf(
    out, sizeof(out),
    "IMU seq=%lu t_us=%llu ok=0 why=%s\n",
    (unsigned long)imuSequence,
    (unsigned long long)esp_timer_get_time(),
    why
  );

  sendImuLine(out, size);
}


void sendImuInfo() {

  char out[200];

  int size = snprintf(
    out, sizeof(out),
    "IMUINFO state=%s addr=0x%02X who=0x%02X hz=%d range=4g,2000dps unit=m/s2,deg/s axes=sensor "
    "good=%lu failed=%lu longest_read_us=%lu\n",
    imuStateName(), (unsigned)imuAddress, (unsigned)imuWhoAmI, 1000 / IMU_PERIOD_MS,
    (unsigned long)imuGoodReads, (unsigned long)imuFailedReads, imuLongestReadUs
  );

  sendLine(out, size);
}


// Called once per loop pass, after the motors have been set.
// Makes AT MOST ONE I2C transaction per call.

void imuService(unsigned long now) {

  if (!imuBusOk) {

    // I2C could not be started at all: say so twice a second.

    if (now - lastImuAttempt >= 500) {

      lastImuAttempt = now;
      imuSequence++;
      sendImuInvalid("NOT_FOUND");
    }

    return;
  }

  if (imuState == IMU_RUNNING) {

    if (now - lastImuSample < IMU_PERIOD_MS) {
      return;
    }

    lastImuSample = now;
    imuSequence++;

    // OUTX_L_G .. OUTZ_H_XL: gyro x, y, z then accel x, y, z, low byte first.

    uint8_t raw[12];

    unsigned long long sampledAt = (unsigned long long)esp_timer_get_time();
    unsigned long started = micros();

    bool ok = imuReadRegisters(imuAddress, 0x22, raw, sizeof(raw));

    unsigned long took = micros() - started;

    if (took > imuLongestReadUs) {
      imuLongestReadUs = took;
    }

    if (!ok) {

      imuFailedReads++;
      imuConsecutiveFails++;

      sendImuInvalid("READ_FAILED");

      if (imuConsecutiveFails >= IMU_FAILS_BEFORE_LOST) {

        imuState = IMU_READ_FAILED;
        lastImuAttempt = now;
      }

      return;
    }

    imuGoodReads++;
    imuConsecutiveFails = 0;

    int16_t counts[6];

    for (int i = 0; i < 6; i++) {
      counts[i] = (int16_t)(((uint16_t)raw[2 * i + 1] << 8) | raw[2 * i]);
    }

    char out[180];

    int size = snprintf(
      out, sizeof(out),
      "IMU seq=%lu t_us=%llu ok=1 ax=%.4f ay=%.4f az=%.4f gx=%.3f gy=%.3f gz=%.3f unit=m/s2,deg/s\n",
      (unsigned long)imuSequence, sampledAt,
      counts[3] * IMU_ACCEL_MS2_PER_COUNT, counts[4] * IMU_ACCEL_MS2_PER_COUNT, counts[5] * IMU_ACCEL_MS2_PER_COUNT,
      counts[0] * IMU_GYRO_DPS_PER_COUNT, counts[1] * IMU_GYRO_DPS_PER_COUNT, counts[2] * IMU_GYRO_DPS_PER_COUNT
    );

    sendImuLine(out, size);
    return;
  }

  if (imuState == IMU_CONFIGURING) {

    // Same three settings as the standalone IMU test, one write per pass.

    static const uint8_t settings[3][2] = {
      {0x12, 0x44},   // CTRL3_C: block data update, register auto-increment
      {0x10, 0x48},   // CTRL1_XL: 104 Hz, +/-4 g
      {0x11, 0x4C},   // CTRL2_G: 104 Hz, +/-2000 deg/s
    };

    if (!imuWriteRegister(imuAddress, settings[imuConfigStep][0], settings[imuConfigStep][1])) {

      imuState = IMU_NOT_FOUND;
      lastImuAttempt = now;
      imuSequence++;
      sendImuInvalid("CONFIG_FAILED");
      return;
    }

    imuConfigStep++;

    if (imuConfigStep >= 3) {

      imuState = IMU_RUNNING;
      imuConsecutiveFails = 0;
      lastImuSample = now;
    }

    return;
  }

  // NOT_FOUND or READ_FAILED: one short WHO_AM_I probe every IMU_RETRY_MS,
  // alternating between the two possible addresses.

  if (now - lastImuAttempt < IMU_RETRY_MS) {
    return;
  }

  lastImuAttempt = now;

  uint8_t address = imuProbeNext ? 0x6B : 0x6A;
  imuProbeNext = !imuProbeNext;

  uint8_t who = 0;

  if (
    imuReadRegisters(address, 0x0F, &who, 1) &&
    (who == 0x69 || who == 0x6A)        // LSM6DS3 or LSM6DS3TR-C
  ) {

    imuAddress = address;
    imuWhoAmI = who;
    imuConfigStep = 0;
    imuState = IMU_CONFIGURING;
    return;
  }

  imuSequence++;
  sendImuInvalid(imuState == IMU_READ_FAILED ? "READ_FAILED" : "NOT_FOUND");
}


'''

SETUP = r'''  // -----------------------------
  // IMU BUS
  // -----------------------------
  //
  // Only starts the bus. Finding and configuring the sensor happens
  // in loop(), one short transaction at a time, so a missing or stuck
  // sensor cannot hold up start-up or the radio.

  imuBusOk = Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN, IMU_I2C_HZ);
  Wire.setTimeOut(IMU_I2C_TIMEOUT_MS);


'''

LOOP = r'''

  // ===================================================
  // IMU TELEMETRY (last: the motors and status are already done)
  // ===================================================

  imuService(now);

  if (now - lastImuInfo >= 1000 && now - lastStatus >= 5 && now - lastChannelLine >= 5 && now - lastNetLine >= 5) {

    lastImuInfo = now;

    sendImuInfo();
  }
}
'''


def build(source_text: str) -> str:
    s = source_text

    def insert_after(anchor: str, addition: str) -> None:
        nonlocal s
        assert s.count(anchor) == 1, f"expected exactly once ({s.count(anchor)}):\n{anchor}"
        s = s.replace(anchor, anchor + addition)

    def insert_before(anchor: str, addition: str) -> None:
        nonlocal s
        assert s.count(anchor) == 1, f"expected exactly once ({s.count(anchor)}):\n{anchor}"
        s = s.replace(anchor, addition + anchor)

    def replace(old: str, new: str) -> None:
        nonlocal s
        assert s.count(old) == 1, f"expected exactly once ({s.count(old)}):\n{old}"
        s = s.replace(old, new)

    insert_after("#include <WiFiUdp.h>\n", "#include <Wire.h>\n#include <esp_timer.h>\n")
    replace("// RADIO / PC HANDOFF BENCH TEST  +  WI-FI PC LINK\n", "// RADIO / PC HANDOFF BENCH TEST  +  WI-FI PC LINK  +  IMU TELEMETRY\n")
    insert_before("// This is ELRS-Handoff-Bench.ino (the USB version tested on the robot)\n",
                  "// GENERATED from ELRS-Handoff-WiFi.ino by scripts/make_telemetry_sketch.py.\n"
                  "// Change handoff or Wi-Fi settings there and regenerate; edit IMU code in\n"
                  "// that script. This build only ADDS read-only IMU telemetry.\n//\n")
    insert_after("#ifndef WIFI_REDUCED_TX_POWER\n#define WIFI_REDUCED_TX_POWER 0\n#endif\n", CONFIG)
    insert_after('static_assert(WIFI_AP_CHANNEL >= 1 && WIFI_AP_CHANNEL <= 13, "WIFI_AP_CHANNEL");\n', ASSERTS)
    insert_after("uint32_t netSignature = 0;\n", STATE)
    insert_before("// =====================================================\n// SETUP\n", FUNCTIONS)
    insert_before("  // A different token every boot, so a PC command from before a\n", SETUP)
    replace("HELLO fw=autobot-handoff-wifi-1 ", "HELLO fw=autobot-handoff-telemetry-1 ")
    replace("    sendNet(now);\n  }\n}\n", "    sendNet(now);\n  }" + LOOP)
    return s


def main() -> int:
    text = build(SOURCE.read_text(encoding="utf-8"))
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    print(f"wrote {TARGET} ({len(text.splitlines())} lines) from {SOURCE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
