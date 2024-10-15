/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stddef.h>
#include <cstdint>
namespace rapidudf {
class Bit {
 public:
  explicit Bit(bool v) : val_(v) {}
  Bit operator+(const Bit& other) const { return *this; }
  Bit operator-(const Bit& other) const { return *this; }
  Bit operator*(const Bit& other) const { return *this; }
  Bit operator/(const Bit& other) const { return *this; }
  Bit operator%(const Bit& other) const { return *this; }
  Bit operator==(const Bit& other) const { return Bit(val_ == other.val_); }
  Bit operator==(bool other) const { return Bit(val_ == other); }
  Bit operator!=(const Bit& other) const { return Bit(val_ != other.val_); }
  Bit operator>=(const Bit& other) const { return Bit(val_ >= other.val_); }
  Bit operator<=(const Bit& other) const { return Bit(val_ <= other.val_); }
  Bit operator>(const Bit& other) const { return Bit(val_ > other.val_); }
  Bit operator<(const Bit& other) const { return Bit(val_ < other.val_); }
  Bit operator&&(const Bit& other) const { return Bit(val_ && other.val_); }
  Bit operator||(const Bit& other) const { return Bit(val_ || other.val_); }
  operator bool() { return val_; }

 private:
  bool val_ = false;
};

inline uint8_t bit_set(uint8_t number, uint8_t n) { return number | ((uint8_t)1 << n); }
inline uint8_t bit_clear(uint8_t number, uint8_t n) { return number & ~((uint8_t)1 << n); }
inline size_t get_bits_byte_size(size_t n) { return (n + 7) / 8 + 8; }
inline size_t get_arena_element_size(size_t n, size_t lanes) {
  size_t rest = n % lanes;
  if (rest == 0) {
    return n;
  }
  return n + lanes - rest;
}
inline uint64_t bits64_set(uint64_t bits, uint8_t i) { return bits |= (1ULL << i); }
inline uint64_t bits64_clear(uint64_t bits, uint8_t i) { return bits &= ~(1ULL << i); }

}  // namespace rapidudf