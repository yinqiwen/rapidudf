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
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

#include "rapidudf/context/context.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/simd/vector.h"

namespace rapidudf {
namespace simd {
class Column {
 public:
  explicit Column(Context& ctx, Vector<double> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_F64){};
  explicit Column(Context& ctx, Vector<float> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_F32){};
  explicit Column(Context& ctx, Vector<StringView> data)
      : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_STRING_VIEW){};
  explicit Column(Context& ctx, Vector<uint64_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_U64){};
  explicit Column(Context& ctx, Vector<int64_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_I64){};
  explicit Column(Context& ctx, Vector<uint32_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_U32){};
  explicit Column(Context& ctx, Vector<int32_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_I32){};
  explicit Column(Context& ctx, Vector<uint16_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_U16){};
  explicit Column(Context& ctx, Vector<int16_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_I16){};
  explicit Column(Context& ctx, Vector<uint8_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_U8){};
  explicit Column(Context& ctx, Vector<int8_t> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_I8){};
  explicit Column(Context& ctx, Vector<Bit> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_BIT){};
  explicit Column(Context& ctx, Vector<Pointer> data) : ctx_(ctx), data_(data.RawData()), element_dtype(DATA_POINTER){};

  template <typename T>
  static Column* FromVector(Context& ctx, Vector<T> data) {
    return ctx.New<simd::Column>(ctx, data);
  }

  Context& GetContext() { return ctx_; }

  bool TypeEquals(const Column& other) const { return element_dtype == other.element_dtype; }

  //   template <typename T>
  //   bool Is() const {
  //     return std::holds_alternative<Vector<T>>(data_);
  //   }
  //   bool IsBit() const { return Is<Bit>(); }

  /**
  ** member methods
  */
  size_t size() const;
  Column* clone();
  Column* take(size_t n);
  Column* filter(Column* bits);
  Column* gather(Column* indices);

  template <typename T>
  absl::StatusOr<Vector<T>> ToVector() const {
    return absl::UnimplementedError("ToVector");
  }

 private:
  Context& ctx_;
  VectorData data_;
  uint32_t element_dtype;
};
}  // namespace simd
}  // namespace rapidudf