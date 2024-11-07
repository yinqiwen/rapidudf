/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
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

#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/column.h"
#include "rapidudf/vector/row.h"
#include "rapidudf/vector/vector.h"
namespace rapidudf {

namespace simd {

class TableSchema;
class Table : public DynObject {
 private:
  struct Deleter {
    void operator()(Table* ptr) const;
  };

 public:
  using SmartPtr = std::unique_ptr<Table, Deleter>;

 public:
  static uint32_t GetIdxByOffset(uint32_t offset);

  template <typename T>
  absl::Status Set(const std::string& name, T&& v) {
    return DoSet(name, std::forward<T>(v));
  }

  template <typename T>
  auto Get(const std::string& name) {
    if constexpr (std::is_same_v<bool, T>) {
      return GetColumn<Bit>(name);
    } else {
      return GetColumn<T>(name);
    }
  }

  VectorData GetColumnByOffset(uint32_t offset);

  /**
  ** Unload loaded column(defined by protobuf/flatbuffers/struct)
  */
  absl::Status UnloadColumn(const std::string& name);
  /**
   ** Unload all loaded column(defined by protobuf/flatbuffers/struct)
   */
  absl::Status UnloadAllColumns();

  template <typename T>
  absl::Status AddRows(const std::vector<const T*>& rows) {
    if (rows.empty()) {
      RUDF_LOG_RETURN_FMT_ERROR("Empty rows to add");
    }
    std::vector<const uint8_t*> objs;
    objs.reserve(rows.size());
    for (auto v : rows) {
      objs.emplace_back(reinterpret_cast<const uint8_t*>(v));
    }
    RowSchema row_schema = GetRowSchema(rows[0]);
    return AddRows(std::move(objs), row_schema);
  }
  template <typename T>
  absl::Status AddRows(const std::vector<T>& rows) {
    if (rows.empty()) {
      RUDF_LOG_RETURN_FMT_ERROR("Empty rows to add");
    }
    std::vector<const uint8_t*> objs;
    objs.reserve(rows.size());
    for (auto& v : rows) {
      objs.emplace_back(reinterpret_cast<const uint8_t*>(&v));
    }
    RowSchema row_schema = GetRowSchema(&rows[0]);
    return AddRows(std::move(objs), row_schema);
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

  Table* Filter(Vector<Bit> bits);
  template <typename T>
  Table* OrderBy(Vector<T> by, bool descending);
  template <typename T>
  Table* Topk(Vector<T> by, uint32_t k, bool descending);
  Table* Take(uint32_t k);
  template <typename T>
  absl::Span<Table*> GroupBy(Vector<T> by);
  absl::Span<Table*> GroupBy(StringView column);

  /**
  ** return column count
  */
  size_t Size() const;
  /**
   ** return row count
   */
  size_t Count() const;

 private:
  Table(Context& ctx, const DynObjectSchema* s) : DynObject(s), ctx_(ctx) {}
  Table* Clone();
  std::vector<int32_t> GetIndices();
  void SetIndices(std::vector<int32_t>&& indices);

  const TableSchema* GetTableSchema() const { return reinterpret_cast<const TableSchema*>(schema_); }

  absl::Status AddRows(std::vector<const uint8_t*>&& rows, const RowSchema& schema);

  absl::StatusOr<uint32_t> GetColumnOffset(const std::string& name);

  bool IsColumnLoaded(uint32_t offset);
  uint8_t* GetColumnMemory(uint32_t offset, const DType& dtype);
  size_t GetColumnMemorySize(const DType& dtype);
  void SetColumnSize(const reflect::Column& column, void* p);

  template <typename T>
  absl::StatusOr<Vector<T>> GetColumn(const std::string& name) {
    uint32_t offset = 0;
    auto result = DynObject::Get<Vector<T>>(name, &offset);
    if (!result.ok()) {
      return result;
    }
    Vector<T> vec = std::move(result.value());
    if (vec.Data() == nullptr) {
      return Vector<T>(GetColumnByOffset(offset));
    }
    return vec;
  }

  template <typename T>
  RowSchema GetRowSchema(const T* obj) {
    if constexpr (std::is_base_of_v<::google::protobuf::Message, T>) {
      const ::google::protobuf::Descriptor* desc = obj->GetDescriptor();
      return RowSchema(desc);
    } else if constexpr (std::is_base_of_v<::flatbuffers::Table, T>) {
      auto* type_table = T::MiniReflectTypeTable();
      return RowSchema(type_table);
    } else {
      return RowSchema(get_dtype<T>());
    }
  }

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

  template <typename T>
  absl::Span<Table*> GroupBy(const T* by, size_t n);

  Table* SubTable(std::vector<int32_t>& indices);

  void SetColumn(uint32_t offset, VectorData vec);

  absl::StatusOr<VectorData> GatherField(uint8_t* vec_ptr, const DType& dtype, Vector<int32_t> indices);

  template <typename T>
  absl::Status LoadProtobufColumn(const Vector<Pointer>& pb_vector, const reflect::Column& column);
  template <typename T>
  absl::Status LoadFlatbuffersColumn(const Vector<Pointer>& fbs_vector, const reflect::Column& column);
  template <typename T>
  absl::Status LoadStructColumn(const Vector<Pointer>& struct_vector, const reflect::Column& column);

  template <typename T>
  absl::Status LoadColumn(const Vector<Pointer>& objs, const reflect::Column& column);

  absl::Status LoadColumn(const Rows& rows, const reflect::Column& column);

  Context& ctx_;
  std::vector<int32_t> indices_;
  std::vector<Rows> rows_;
  friend class TableSchema;
};

}  // namespace simd
}  // namespace rapidudf