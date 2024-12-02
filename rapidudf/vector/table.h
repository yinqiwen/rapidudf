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

#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/column.h"
#include "rapidudf/vector/row.h"
#include "rapidudf/vector/vector.h"
namespace rapidudf {

namespace exec {
template <class R, class... Args>
absl::StatusOr<R> eval_function(const std::string& source, Args... args);

template <class R, class... Args>
absl::StatusOr<R> eval_expression(const std::string& source, const std::vector<std::string>& arg_names, Args... args);
}  // namespace exec

namespace simd {

class TableSchema;
class Table : public DynObject {
 private:
  struct Deleter {
    void operator()(Table* ptr) const;
  };

  using PartialRows = std::pair<const RowSchema*, std::vector<const uint8_t*>>;
  using PartialRow = std::pair<const RowSchema*, const uint8_t*>;

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
  void UnloadAllColumns();

  template <typename... T>
  absl::Status AddRows(const std::vector<T>&... rows) {
    std::vector<absl::StatusOr<PartialRows>> add_results;
    std::vector<PartialRows> add_rows;
    add_results.emplace_back((GetPartialRows(rows), ...));

    for (auto& result : add_results) {
      if (!result.ok()) {
        return result.status();
      }
      add_rows.emplace_back(std::move(result.value()));
    }

    return DoAddRows(std::move(add_rows));
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

  template <typename... T>
  absl::Status InsertRow(size_t pos, const T*... row) {
    std::vector<absl::StatusOr<PartialRow>> add_results;
    add_results.emplace_back((GetPartialRow(row), ...));
    std::vector<PartialRow> insert_row;
    for (auto& result : add_results) {
      if (!result.ok()) {
        return result.status();
      }
      insert_row.emplace_back(std::move(result.value()));
    }
    return InsertRow(pos, insert_row);
  }

  template <typename... T>
  absl::Status AppendRow(const T*... row) {
    size_t pos = Count();
    return InsertRow(pos, row...);
  }

  template <typename T>
  const T* GetRow(size_t idx) {
    const RowSchema* schema = GetRowSchema<T>();
    if (schema == nullptr) {
      THROW_LOGIC_ERR("Invalid row object type to get schema");
    }
    int row_idx = GetRowIdx(*schema);
    if (row_idx < 0) {
      THROW_LOGIC_ERR("No row found for schema");
    }
    size_t count = Count();
    if (idx >= count) {
      THROW_LOGIC_ERR("GetRow outofbound with idx:{} size:{}", idx, count);
    }
    return rows_[row_idx].GetRowPtrs()[idx].As<T>();
  }
  std::pair<Table*, Table*> Split(Vector<Bit> bits);
  Table* Filter(Vector<Bit> bits);
  Table* OrderBy(StringView column, bool descending);
  template <typename T>
  Table* OrderBy(Vector<T> by, bool descending);
  template <typename T>
  Table* Topk(Vector<T> by, uint32_t k, bool descending);
  Table* Head(uint32_t k);
  Table* Tail(uint32_t k);
  template <typename T>
  absl::Span<Table*> GroupBy(Vector<T> by);
  absl::Span<Table*> GroupBy(StringView column);

  Vector<Bit> Dedup(StringView column, uint32_t k);
  template <typename T, bool restart = true>
  inline Vector<Bit> Select(std::function<int(const T*, size_t)>&& select, Vector<Bit>* mask = nullptr) {
    const RowSchema* schema = GetRowSchema<T>();
    if (schema == nullptr) {
      THROW_LOGIC_ERR("Invalid row object type to get schema");
    }
    int row_idx = GetRowIdx(*schema);
    if (row_idx < 0) {
      THROW_LOGIC_ERR("No row found for schema");
    }
    size_t count = Count();
    Vector<Bit> select_mask;
    size_t non_skip_start = 0;
    if (mask != nullptr) {
      select_mask = *mask;
      for (size_t i = 0; i < count; i++) {
        if (select_mask[i]) {
          non_skip_start = i;
          break;
        }
      }
    } else {
      select_mask = ctx_.NewSimdVector<Bit>(count);
      memset(select_mask.GetVectorData().MutableData<uint8_t>(), 0, select_mask.BytesCapacity());
    }

    for (size_t i = non_skip_start; i < count; i++) {
      if (select_mask[i]) {
        continue;
      }
      T* row = rows_[row_idx].GetRowPtrs()[i].As<T>();
      int rc = select(row, i);
      if (rc == 1) {
        // skip[i] = true;
        select_mask.Set(i, Bit(true));
        if constexpr (restart) {
          if (i == non_skip_start) {
            non_skip_start++;
          }
          i = non_skip_start;  // restart
        }
      } else if (rc == -1) {
        break;
      }
    }
    return select_mask;
  }

  /**
  ** return column count
  */
  size_t Size() const;
  /**
   ** return row count
   */
  size_t Count() const;

  template <class R, class... Args>
  absl::StatusOr<R> EvalFunction(const std::string& source, Args... args) {
    return exec::eval_function<R, Args...>(source, args...);
  }

  template <class R, class... Args>
  absl::StatusOr<R> EvalExpression(const std::string& source, const std::vector<std::string>& arg_names, Args... args) {
    return exec::eval_expression<R, Args...>(source, arg_names, args...);
  }

 private:
  Table(Context& ctx, const DynObjectSchema* s);
  Table* Clone();
  std::vector<int32_t> GetIndices();
  void SetIndices(std::vector<int32_t>&& indices);

  const TableSchema* GetTableSchema() const { return reinterpret_cast<const TableSchema*>(schema_); }

  absl::Status DoAddRows(std::vector<PartialRows>&& rows);
  absl::Status DoAddRows(std::vector<const uint8_t*>&& rows, const RowSchema& schema);
  absl::Status InsertRow(size_t pos, const std::vector<PartialRow>& row);

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
  const RowSchema* GetRowSchema() {
    if constexpr (std::is_base_of_v<::google::protobuf::Message, T>) {
      static T msg;
      const ::google::protobuf::Descriptor* desc = msg.GetDescriptor();

      return GetRowSchema(RowSchema(desc));
    } else if constexpr (std::is_base_of_v<::flatbuffers::Table, T>) {
      auto* type_table = T::MiniReflectTypeTable();
      return GetRowSchema(RowSchema(type_table));
    } else {
      return GetRowSchema(RowSchema(get_dtype<T>()));
    }
  }
  template <typename T>
  absl::StatusOr<PartialRows> GetPartialRows(const std::vector<T>& rows) {
    if (rows.empty()) {
      RUDF_LOG_RETURN_FMT_ERROR("Empty rows to add");
    }
    PartialRows pratial_rows;
    pratial_rows.second.reserve(rows.size());
    for (auto& v : rows) {
      if constexpr (std::is_pointer_v<T>) {
        pratial_rows.second.emplace_back(reinterpret_cast<const uint8_t*>(v));
      } else {
        pratial_rows.second.emplace_back(reinterpret_cast<const uint8_t*>(&v));
      }
    }
    const RowSchema* schema = nullptr;
    if constexpr (std::is_pointer_v<T>) {
      schema = GetRowSchema<std::decay_t<std::remove_pointer_t<T>>>();
    } else {
      schema = GetRowSchema<T>();
    }

    if (schema == nullptr) {
      RUDF_LOG_RETURN_FMT_ERROR("Invalid row object type to get schema");
    }
    pratial_rows.first = schema;
    return pratial_rows;
  }

  template <typename T>
  absl::StatusOr<PartialRow> GetPartialRow(const T* row) {
    PartialRow partial_row;
    const RowSchema* schema = GetRowSchema<T>();
    if (schema == nullptr) {
      RUDF_LOG_RETURN_FMT_ERROR("Invalid row object type to get schema");
    }
    partial_row.second = reinterpret_cast<const uint8_t*>(row);
    partial_row.first = schema;
    return partial_row;
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

  const RowSchema* GetRowSchema(const RowSchema& schema);
  int GetRowIdx(const RowSchema& schema);

  template <typename T>
  Vector<Bit> Dedup(const T* data, size_t n, size_t k);

  Context& ctx_;
  std::vector<int32_t> indices_;
  std::vector<Rows> rows_;
  friend class TableSchema;
};

}  // namespace simd
}  // namespace rapidudf