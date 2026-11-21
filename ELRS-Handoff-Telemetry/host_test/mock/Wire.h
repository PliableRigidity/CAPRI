// Host-side stand-in for the ESP32 core's Wire (I2C) with a simulated LSM6DS3 on the bus.
// The other mock headers are shared with ELRS-Handoff-WiFi/host_test/mock.
//
// Modelled on the core: endTransmission(false) only queues the register address; the bus
// transaction happens in requestFrom() (one combined write-then-read) or in endTransmission(true).
// Every bus transaction is counted, so a test can show that the sketch never makes more than one
// per loop pass. A hung bus costs one timeout per transaction, visible through micros().
#pragma once

#include "Arduino.h"

namespace mock {

enum ImuMode { IMU_PRESENT, IMU_ABSENT, IMU_HUNG };
inline ImuMode imu_mode = IMU_PRESENT;
inline uint8_t imu_address = 0x6B;          // as found on the robot by the standalone IMU test
inline uint8_t imu_who = 0x69;
inline int16_t imu_gyro[3] = {0, 0, 0};     // raw counts, 70 mdps per count
inline int16_t imu_accel[3] = {0, 0, 8197}; // raw counts, 0.122 mg per count: +1 g on sensor z
inline uint8_t imu_written[256] = {0};      // last value written to each register
inline bool wire_begin_fails = false;
inline int wire_sda = -1, wire_scl = -1, wire_timeout_ms = -1;
inline unsigned long wire_hz = 0;
inline unsigned long i2c_transactions = 0, i2c_timeouts = 0;
inline int i2c_this_pass = 0, i2c_most_per_pass = 0;
inline unsigned long i2c_busy_us = 0;       // bus time used in the current loop pass

inline uint8_t imu_register(uint8_t reg) {
  if (reg == 0x0F) return imu_who;
  if (reg >= 0x22 && reg < 0x2E) {
    const int index = (reg - 0x22) / 2;
    const uint16_t value = static_cast<uint16_t>(index < 3 ? imu_gyro[index] : imu_accel[index - 3]);
    return static_cast<uint8_t>(((reg - 0x22) % 2) ? (value >> 8) : (value & 0xFF));
  }
  return imu_written[reg];
}

inline bool i2c_transaction(uint8_t address) {
  ++i2c_transactions;
  ++i2c_this_pass;
  i2c_most_per_pass = std::max(i2c_most_per_pass, i2c_this_pass);
  if (imu_mode == IMU_HUNG) {
    ++i2c_timeouts;
    i2c_busy_us += 1000UL * static_cast<unsigned long>(std::max(0, wire_timeout_ms));
    return false;
  }
  if (imu_mode == IMU_ABSENT || address != imu_address) {
    i2c_busy_us += 120;                      // address byte, no acknowledge
    return false;
  }
  i2c_busy_us += 1600;                       // a 12-byte read at 100 kHz
  return true;
}

}  // namespace mock

inline unsigned long micros() { return mock::now_ms * 1000UL + mock::i2c_busy_us; }

class TwoWire {
 public:
  bool begin(int sda, int scl, uint32_t hz) {
    mock::wire_sda = sda;
    mock::wire_scl = scl;
    mock::wire_hz = hz;
    return !mock::wire_begin_fails;
  }
  void setTimeOut(uint16_t ms) { mock::wire_timeout_ms = ms; }

  void beginTransmission(uint8_t address) {
    address_ = address;
    tx_.clear();
  }
  size_t write(uint8_t value) {
    tx_.push_back(value);
    return 1;
  }
  uint8_t endTransmission(bool stop = true) {
    if (!stop) return 0;                     // queued; the bus is used by requestFrom()
    if (!mock::i2c_transaction(address_)) return mock::imu_mode == mock::IMU_HUNG ? 5 : 2;
    if (tx_.size() >= 2) mock::imu_written[tx_[0]] = tx_[1];
    return 0;
  }
  uint8_t requestFrom(uint8_t address, uint8_t count) {
    rx_.clear();
    if (!mock::i2c_transaction(address)) return 0;
    const uint8_t first = tx_.empty() ? 0 : tx_[0];
    for (uint8_t i = 0; i < count; ++i) rx_.push_back(mock::imu_register(static_cast<uint8_t>(first + i)));
    return count;
  }
  int read() {
    if (rx_.empty()) return -1;
    const int value = rx_.front();
    rx_.pop_front();
    return value;
  }

 private:
  uint8_t address_ = 0;
  std::vector<uint8_t> tx_;
  std::deque<uint8_t> rx_;
};
inline TwoWire Wire;
