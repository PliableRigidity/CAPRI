// Host-side stand-in for the ESP32 core's WiFi object: just the access-point calls the sketch makes.
// The "network" is two queues in namespace mock; tests put datagrams in and read what came out.
#pragma once

#include "Arduino.h"

#define WIFI_AP 2
#define WIFI_POWER_8_5dBm 34

namespace mock {

inline int tx_power = -1;                // -1 = left at the core's default

struct Datagram {
  IPAddress ip;
  uint16_t port = 0;
  std::string data;
};

inline std::deque<Datagram> wifi_in;     // laptop -> robot, waiting in the socket
inline std::vector<Datagram> wifi_out;   // robot -> laptop
inline bool ap_fails = false;            // softAP() returns false
inline bool ap_started = false;
inline std::string ap_ssid, ap_password;
inline int ap_channel = 0, ap_hidden = -1, ap_max_connections = 0;
inline int stations = 0;                 // laptops joined
inline bool modem_sleep = true;
inline int udp_port = 0;

}  // namespace mock

class WiFiClass {
 public:
  static void persistent(bool) {}
  bool mode(int) { return true; }
  bool softAP(const char* ssid, const char* password, int channel, int hidden, int max_connections) {
    mock::ap_ssid = ssid;
    mock::ap_password = password ? password : "";
    mock::ap_channel = channel;
    mock::ap_hidden = hidden;
    mock::ap_max_connections = max_connections;
    mock::ap_started = !mock::ap_fails;
    return mock::ap_started;
  }
  bool setSleep(bool enabled) {
    mock::modem_sleep = enabled;
    return true;
  }
  bool setTxPower(int power) {
    mock::tx_power = power;
    return true;
  }
  uint8_t softAPgetStationNum() { return static_cast<uint8_t>(mock::stations); }
  IPAddress softAPIP() { return IPAddress(192, 168, 4, 1); }
};
inline WiFiClass WiFi;
