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

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"

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
  absl::Status BuildFromProtobufVector(const std::vector<T>& pb_vector) {
    std::vector<const ::google::protobuf::Message*> msg_vector;
    msg_vector.reserve(pb_vector.size());
    for (auto& msg : pb_vector) {
      msg_vector.emplace_back(&msg);
    }
    return BuildFromProtobufVector(msg_vector);
  }

  template <typename T>
  absl::Status BuildFromProtobufVector(const std::vector<T*>& pb_vector) {
    std::vector<const ::google::protobuf::Message*> msg_vector;
    msg_vector.reserve(pb_vector.size());
    for (auto msg : pb_vector) {
      msg_vector.emplace_back(msg);
    }
    return BuildFromProtobufVector(msg_vector);
  }
  template <typename T>
  absl::Status BuildFromProtobufVector(const std::vector<const T*>& pb_vector) {
    std::vector<const ::google::protobuf::Message*> msg_vector;
    msg_vector.reserve(pb_vector.size());
    for (auto msg : pb_vector) {
      msg_vector.emplace_back(msg);
    }
    return BuildFromProtobufVector(msg_vector);
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
  void SetSize(uint32_t k);

  absl::Status BuildFromProtobufVector(const std::vector<const ::google::protobuf::Message*>& pb_vector);

  template <typename T>
  absl::Status SetColumnByProtobufField(const std::vector<const ::google::protobuf::Message*>& pb_vector,
                                        const ::google::protobuf::Reflection* reflect,
                                        const ::google::protobuf::FieldDescriptor* field);

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
  absl::Status BuildFromProtobuf() {
    static T msg;
    return BuildFromProtobuf(&msg);
  }

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
  bool ExistColumn(const std::string& name) const;

 private:
  TableSchema(const std::string& name, size_t reserved_size) : DynObjectSchema(name, reserved_size, Flags(true)) {}
  template <typename T>
  absl::Status AddField(const std::string& name) {
    return absl::UnimplementedError("AddField");
  }
  typename DynObject::SmartPtr NewObject() const { return nullptr; }

  absl::Status BuildFromProtobuf(const ::google::protobuf::Message* msg);
};
}  // namespace simd
}  // namespace rapidudf