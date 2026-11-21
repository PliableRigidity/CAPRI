// Host-side stand-in: microseconds since boot, 64-bit (does not wrap), from the mock clock.
#pragma once

#include "Wire.h"

inline int64_t esp_timer_get_time() { return static_cast<int64_t>(mock::now_ms) * 1000 + static_cast<int64_t>(mock::i2c_busy_us); }
