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
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace simd {
class TableSchema;
class Table : public DynObject {
 private:
  struct Deleter {
    void operator()(Table* ptr) {
      ptr->~Table();
      uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
      delete[] bytes;
    }
  };

 public:
  typedef std::unique_ptr<Table, Deleter> SmartPtr;

 public:
  template <typename T>
  absl::Status Set(const std::string& name, T&& v) {
    return DoSet(name, std::forward<T>(v));
  }

  template <template <class, class> class Map, template <class> class Vec, class V>
  absl::Status AddMap(Map<std::string, Vec<V>>&& values) {
    for (auto& [name, v] : values) {
      auto status = DoSet(name, std::move(v));
      if (!status.ok()) {
        return status;
      }
    }
    std::ignore = ctx_.New<Map<std::string, Vec<V>>>(std::move(values));
    return absl::OkStatus();
  }

  template <typename T>
  auto Get(const std::string& name) {
    if constexpr (std::is_same_v<bool, T>) {
      return DynObject::Get<Vector<Bit>>(name);
    } else {
      return DynObject::Get<Vector<T>>(name);
    }
  }

  Table* Filter(Vector<Bit> bits);
  template <typename T>
  Table* OrderBy(Vector<T> by, bool descending);
  template <typename T>
  Table* Topk(Vector<T> by, uint32_t k, bool descending);
  Table* Take(uint32_t k);
  size_t Size() const;

 private:
  Table(Context& ctx, const DynObjectSchema* s) : DynObject(s), ctx_(ctx) {}

  Table* Clone();
  Vector<int32_t> GetIndices();

  template <typename T>
  absl::Status DoSet(const std::string& name, const std::vector<T>& v) {
    auto vec = ctx_.NewSimdVector(v);
    return DynObject::DoSet(name, std::move(vec));
  }
  template <typename T>
  absl::Status DoSet(const std::string& name, std::vector<T>&& v) {
    auto vec = ctx_.NewSimdVector(v);
    std::ignore = ctx_.New<std::vector<T>>(std::move(v));
    return DynObject::DoSet(name, std::move(vec));
  }

  void SetColumn(uint32_t offset, VectorData vec);

  Context& ctx_;
  Vector<int32_t> indices_;
  friend class TableSchema;
};

class TableSchema : public DynObjectSchema {
 public:
  using InitFunc = std::function<void(TableSchema* s)>;
  static const TableSchema* GetOrCreate(const std::string& name, InitFunc&& init);
  static const TableSchema* Get(const std::string& name);
  static TableSchema* GetMutable(const std::string& name);

  typename Table::SmartPtr NewTable(Context& ctx) const;

  template <typename T>
  absl::Status AddColumn(const std::string& name) {
    if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      return DynObjectSchema::Add(name, get_dtype<simd::Vector<StringView>>());
    } else if constexpr (std::is_same_v<bool, T>) {
      return DynObjectSchema::Add(name, get_dtype<simd::Vector<Bit>>());
    } else {
      return Add(name, get_dtype<simd::Vector<T>>());
    }
  }

 private:
  TableSchema(const std::string& name, size_t reserved_size) : DynObjectSchema(name, reserved_size, Flags(true)) {}
  template <typename T>
  absl::Status AddField(const std::string& name) {
    return absl::UnimplementedError("AddField");
  }
  typename DynObject::SmartPtr NewObject() const { return nullptr; }
};
}  // namespace simd
}  // namespace rapidudf