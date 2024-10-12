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
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

#include "rapidudf/context/context.h"
#include "rapidudf/types/simd/column.h"

namespace rapidudf {
namespace simd {
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
  ColumnTable column_table_;
  Vector<int32_t> indices_;
};
}  // namespace simd
}  // namespace rapidudf