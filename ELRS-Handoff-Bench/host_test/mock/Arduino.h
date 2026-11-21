// Minimal host-side stand-in for the ESP32 Arduino core (3.3.0), just enough
// to compile ELRS-Handoff-Bench.ino and ELRS-Control.ino unmodified on a PC.
// Arduino builds ignore this folder (only the sketch root and src/ are built).
//
// The pin model follows the core source, not a measurement:
//   - analogWrite attaches the pin to LEDC, starting at duty 0
//     (esp32-hal-ledc.c, analogWrite -> ledcAttach -> ledcWrite);
//   - digitalWrite on a pin that is not a GPIO is ignored
//     (esp32-hal-gpio.c, __digitalWrite: "IO %i is not set as GPIO");
//   - analogWrite(pin, 255) at 8 bits is constant HIGH (ledcWrite: max duty -> full on);
//   - pinMode takes the pin back from LEDC.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

using std::abs;
using std::max;
using std::min;

#define constrain(amt, low, high) ((amt) < (low) ? (low) : ((amt) > (high) ? (high) : (amt)))

#define HIGH 1
#define LOW 0
#define OUTPUT 3
#define SERIAL_8N1 0x800001c
#define ARDUINO_USB_CDC_ON_BOOT 1

namespace mock {

inline unsigned long now_ms = 0;

enum Bus { BUS_NONE, BUS_GPIO, BUS_LEDC };
struct Pin {
  Bus bus = BUS_NONE;
  int level = 0;
  int duty = 0;  // out of 256
};
inline Pin pins[22];
inline unsigned long ignored_digital_writes = 0;  // digitalWrite on a non-GPIO pin
inline unsigned long digital_writes = 0;          // every digitalWrite call
inline void (*on_delay)(unsigned long) = nullptr; // called when the sketch calls delay()

// What the pin is actually driving, as a duty out of 256.
inline int pinOut(int pin) {
  const Pin& p = pins[pin];
  if (p.bus == BUS_GPIO) return p.level ? 256 : 0;
  if (p.bus == BUS_LEDC) return p.duty;
  return -1;  // never configured: floating
}

inline void resetBoard() {
  for (Pin& p : pins) p = Pin();
  ignored_digital_writes = 0;
  digital_writes = 0;
}

}  // namespace mock

inline unsigned long millis() { return mock::now_ms; }
inline void delay(unsigned long ms) {
  if (mock::on_delay) mock::on_delay(ms);
  mock::now_ms += ms;
}

inline void pinMode(uint8_t pin, uint8_t) { mock::pins[pin] = {mock::BUS_GPIO, 0, 0}; }

inline void digitalWrite(uint8_t pin, uint8_t value) {
  ++mock::digital_writes;
  if (mock::pins[pin].bus == mock::BUS_GPIO) {
    mock::pins[pin].level = value ? 1 : 0;
  } else {
    ++mock::ignored_digital_writes;  // core 3.x logs an error and does nothing
  }
}

inline void analogWrite(uint8_t pin, int value) {
  if (mock::pins[pin].bus != mock::BUS_LEDC) mock::pins[pin] = {mock::BUS_LEDC, 0, 0};
  mock::pins[pin].duty = value == 255 ? 256 : value;
}

inline long map(long x, long in_min, long in_max, long out_min, long out_max) {
  const long run = in_max - in_min;
  if (run == 0) return -1;
  const long rise = out_max - out_min;
  const long delta = x - in_min;
  return (delta * rise) / run + out_min;
}

// USB Serial/JTAG port.
class HWCDC {
 public:
  void begin(unsigned long baud = 0) { begun_baud = baud; }
  int available() { return static_cast<int>(rx.size()); }
  int read() {
    if (rx.empty()) return -1;
    const int value = rx.front();
    rx.pop_front();
    return value;
  }
  int availableForWrite() { return tx_space; }
  size_t write(const uint8_t* data, size_t size) {
    tx.append(reinterpret_cast<const char*>(data), size);
    return size;
  }

  // Text output, only used by the original RadioMaster sketch.
  void print(const char* text) { tx += text; }
  template <typename T>
  void print(T value) { tx += std::to_string(static_cast<long long>(value)); }
  void println() { tx += "\n"; }
  template <typename T>
  void println(T value) {
    print(value);
    tx += "\n";
  }

  std::deque<uint8_t> rx;  // PC -> ESP
  std::string tx;          // ESP -> PC
  int tx_space = 256;      // set to 0 to model a PC that is not reading
  unsigned long begun_baud = 0;
};
inline HWCDC HWCDCSerial;
#define Serial HWCDCSerial

// UART (the ELRS receiver).
class HardwareSerial {
 public:
  explicit HardwareSerial(int number) : number(number) {}
  void begin(unsigned long baud, uint32_t, int rx_pin, int tx_pin) {
    begun_baud = baud;
    begun_rx = rx_pin;
    begun_tx = tx_pin;
  }
  int available() { return static_cast<int>(rx.size()); }
  int read() {
    if (rx.empty()) return -1;
    const int value = rx.front();
    rx.pop_front();
    return value;
  }

  std::deque<uint8_t> rx;
  int number;
  unsigned long begun_baud = 0;
  int begun_rx = -99;
  int begun_tx = -99;
};
