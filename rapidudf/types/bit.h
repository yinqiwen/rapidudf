/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
}  // namespace rapidudf