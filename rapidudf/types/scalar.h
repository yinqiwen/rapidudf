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
#include <variant>
#include "absl/status/statusor.h"
#include "rapidudf/context/context.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

class Scalar {
 private:
  using Internal = std::variant<StringView, double, float, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                                uint8_t, int8_t, Bit>;

 public:
  explicit Scalar(Context& ctx, double data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, float data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, StringView data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, uint64_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, int64_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, uint32_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, int32_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, uint16_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, int16_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, uint8_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, int8_t data) : ctx_(ctx), data_(data){};
  explicit Scalar(Context& ctx, Bit data) : ctx_(ctx), data_(data){};

  Context& GetContext() { return ctx_; }

  template <typename T>
  absl::StatusOr<T> To() const {
    return std::visit(
        [](auto&& arg) {
          using arg_t = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, arg_t>) {
            return absl::StatusOr<T>(arg);
          } else {
            if constexpr (std::is_arithmetic_v<arg_t> && std::is_arithmetic_v<T>) {
              return absl::StatusOr<T>(static_cast<T>(arg));
            }
            return absl::StatusOr<T>(absl::InvalidArgumentError("invalid type to extract scalar data"));
          }
        },
        data_);
  }
  double to_f64() const;
  float to_f32() const;
  uint64_t to_u64() const;
  uint32_t to_u32() const;
  uint16_t to_u16() const;
  uint8_t to_u8() const;
  int64_t to_i64() const;
  int32_t to_i32() const;
  int16_t to_i16() const;
  int8_t to_i8() const;
  Bit to_bit() const;
  StringView to_string_view() const;

  Internal& GetInternal() { return data_; }

 private:
  template <typename T>
  T ToPrimitive() const {
    auto result = To<T>();
    if (!result.ok()) {
      THROW_LOGIC_ERR(result.status().ToString());
    }
    return result.value();
  }

  Context& ctx_;
  Internal data_;
};

template <typename T>
Scalar* to_scalar(Context& ctx, T v) {
  return ctx.New<Scalar>(ctx, v);
}

}  // namespace rapidudf