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
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/scalar.h"
#include "rapidudf/types/simd_vector.h"
namespace rapidudf {

namespace simd {
class Table;
using TablePtr = std::shared_ptr<Table>;

class Column {
 private:
  using Internal = std::variant<TablePtr, Vector<StringView>, Vector<double>, Vector<float>, Vector<uint64_t>,
                                Vector<int64_t>, Vector<uint32_t>, Vector<int32_t>, Vector<uint16_t>, Vector<int16_t>,
                                Vector<uint8_t>, Vector<int8_t>, Vector<Bit>, Vector<Pointer>>;

 public:
  explicit Column(Context& ctx, Vector<double> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<float> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<StringView> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<uint64_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<int64_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<uint32_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<int32_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<uint16_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<int16_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<uint8_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<int8_t> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<Bit> data) : ctx_(ctx), data_(data){};
  explicit Column(Context& ctx, Vector<Pointer> data) : ctx_(ctx), data_(data){};

  template <typename T>
  static Column* FromVector(Context& ctx, Vector<T> data) {
    return ctx.New<simd::Column>(ctx, data);
  }

  Context& GetContext() { return ctx_; }

  Internal& GetInternal() { return data_; }

  bool TypeEquals(const Column& other) const { return data_.index() == other.data_.index(); }

  template <typename T>
  bool Is() const {
    return std::holds_alternative<Vector<T>>(data_);
  }
  bool IsBit() const { return Is<Bit>(); }

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
    return std::visit(
        [](auto&& arg) {
          using arg_t = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<arg_t, Vector<T>>) {
            return absl::StatusOr<Vector<T>>(arg);
          } else {
            return absl::StatusOr<Vector<T>>(absl::InvalidArgumentError("invalid type to get vector data"));
          }
        },
        data_);
  }

 private:
  Context& ctx_;
  Internal data_;
};

class Table {
 public:
  explicit Table(Context& ctx) : ctx_(ctx) {}
  Context& GetContext() { return ctx_; }
  template <typename T>
  absl::Status Add(const std::string& name, std::vector<T>&& data) {
    auto vec = ctx_.NewSimdVector(data);
    auto p = ctx_.New<Column>(ctx_, vec);
    std::ignore = ctx_.New<std::vector<T>>(std::move(data));
    return Add(name, p);
  }

  template <typename T>
  absl::Status Add(const std::string& name, simd::Vector<T> data) {
    auto p = ctx_.New<Column>(ctx_, data);
    return Add(name, p);
  }

  template <template <class, class> class Map, template <class> class Vec, class V>
  absl::Status AddMap(Map<std::string, Vec<V>>&& values) {
    for (auto& [name, v] : values) {
      auto status = AddColumn(name, v);
      if (!status.ok()) {
        return status;
      }
    }
    std::ignore = ctx_.New<Map<std::string, Vec<V>>>(std::move(values));
    return absl::OkStatus();
  }
  absl::Status Add(const std::string& name, Column* column);
  absl::StatusOr<Column**> Get(StringView name);
  Column* operator[](StringView name);
  Table* Filter(Column* column);
  Table* OrderBy(simd::Table* table, simd::Column* by, bool descending);
  Table* Topk(simd::Table* table, simd::Column* by, uint32_t k, bool descending);
  Table* Take(uint32_t k);

  void Set(const std::string& name, Column* column);
  size_t Size() const;
  Vector<int32_t> GetIndices();

  void Visit(std::function<void(const std::string&, Column*)>&& f);

 private:
  using ColumnTable = absl::flat_hash_map<std::string, Column*>;
  template <typename T>
  absl::Status AddColumn(const std::string& name, std::vector<T>& data) {
    auto vec = ctx_.NewSimdVector(data);
    auto p = ctx_.New<Column>(ctx_, vec);
    return Add(name, p);
  }

  Context& ctx_;
  // ColumnArray columns_;
  ColumnTable column_table_;
  Vector<int32_t> indices_;
};
}  // namespace simd
}  // namespace rapidudf