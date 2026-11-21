// Host-side stand-in for NetworkUDP (WiFiUDP) of the ESP32 core 3.3.0.
// It copies two behaviours of the real class that matter to the sketch (Network/src/NetworkUdp.cpp):
//   - parsePacket() never waits, and returns 0 while a previous datagram still has unread bytes;
//   - clear() throws the rest of the current datagram away.
#pragma once

#include "WiFi.h"

class WiFiUDP {
 public:
  uint8_t begin(uint16_t port) {
    mock::udp_port = port;
    return 1;
  }

  int parsePacket() {
    if (has_buffer_) return 0;
    if (mock::wifi_in.empty()) return 0;
    current_ = mock::wifi_in.front();
    mock::wifi_in.pop_front();
    position_ = 0;
    has_buffer_ = !current_.data.empty();
    return static_cast<int>(current_.data.size());
  }

  int read(uint8_t* buffer, size_t length) {
    if (!has_buffer_) return -1;
    const size_t count = std::min(length, current_.data.size() - position_);
    std::memcpy(buffer, current_.data.data() + position_, count);
    position_ += count;
    if (position_ >= current_.data.size()) has_buffer_ = false;
    return static_cast<int>(count);
  }

  void clear() { has_buffer_ = false; }

  IPAddress remoteIP() { return current_.ip; }
  uint16_t remotePort() { return current_.port; }

  int beginPacket(IPAddress ip, uint16_t port) {
    outgoing_ = mock::Datagram{ip, port, ""};
    return 1;
  }
  size_t write(const uint8_t* data, size_t size) {
    outgoing_.data.append(reinterpret_cast<const char*>(data), size);
    return size;
  }
  int endPacket() {
    mock::wifi_out.push_back(outgoing_);
    return 1;
  }

 private:
  mock::Datagram current_, outgoing_;
  size_t position_ = 0;
  bool has_buffer_ = false;
};
