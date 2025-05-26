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

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"

#include "rapidudf/common/variadic_template_helper.h"
#include "rapidudf/context/context.h"
#include "rapidudf/executors/thread_pool.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/table/column.h"
#include "rapidudf/table/row.h"
#include "rapidudf/table/visitor.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {

namespace exec {
template <class R, class... Args>
absl::StatusOr<R> eval_function(const std::string& source, Args... args);

template <class R, class... Args>
absl::StatusOr<R> eval_expression(const std::string& source, const std::vector<std::string>& arg_names, Args... args);
}  // namespace exec
namespace table {

class TableSchema;
class Table : public DynObject {
 private:
  struct Deleter {
    void operator()(Table* ptr) const;
  };

  using PartialRows = std::pair<const RowSchema*, std::vector<const uint8_t*>>;
  using PartialRow = std::pair<const RowSchema*, const uint8_t*>;
  using DistinctIndiceTable = absl::flat_hash_map<int32_t, std::vector<int32_t>>;

 public:
  using SmartPtr = std::unique_ptr<Table, Deleter>;

 public:
  static uint32_t GetIdxByOffset(uint32_t offset);

  template <typename T>
  auto Get(const std::string& name) {
    if constexpr (std::is_same_v<bool, T> || std::is_same_v<Bit, T>) {
      return GetColumn<Bit>(name);
    } else {
      return GetColumn<T>(name);
    }
  }

  // VectorBase* GetColumnByOffset(uint32_t offset);

  /**
  ** Unload loaded column(defined by protobuf/flatbuffers/struct)
  */
  absl::Status UnloadColumn(const std::string& name);
  /**
   ** Unload all loaded column(defined by protobuf/flatbuffers/struct)
   */
  void UnloadAllColumns();
  template <typename T>
  Vector<T>* LoadColumnByOffset(uint32_t offset);
  VectorBase* LoadColumnBaseByOffset(uint32_t offset);

  template <typename T>
  Vector<T>* GetColumnByOffset(uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
    Vector<T>* vec = nullptr;
    memcpy(&vec, vec_ptr, sizeof(Vector<T>*));
    if (vec != nullptr || vec->Size() >= Count()) {
      // reuse
      return vec;
    } else {
      vec = ctx_.NewVector<T>(Count());
      memcpy(vec_ptr, &vec, sizeof(Vector<T>*));
      return vec;
    }
  }

  template <typename... T>
  absl::Status AddRows(const std::vector<T>&... rows) {
    std::vector<absl::StatusOr<PartialRows>> add_results;
    std::vector<PartialRows> add_rows;
    (add_results.push_back(GetPartialRows(rows)), ...);

    for (auto& result : add_results) {
      if (!result.ok()) {
        return result.status();
      }
      add_rows.emplace_back(std::move(result.value()));
    }

    return DoAddRows(std::move(add_rows));
  }

  template <typename... T>
  absl::Status InsertRow(size_t pos, const T*... row) {
    std::vector<absl::StatusOr<PartialRow>> add_results;
    (add_results.push_back(GetPartialRow(row)), ...);

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
  const T* SlowGetRow(size_t idx) {
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
    return rows_[row_idx].GetRowPtrs()->Value(idx).template As<T>();
    // return rows_[row_idx].GetRowPtrs()[idx].As<T>();
  }

  template <typename R, typename... T>
  absl::Status Foreach(typename VisitorSignatureHelper<R, T...>::type f,
                       std::shared_ptr<ThreadPool> thread_pool = nullptr, Vector<Bit>* mask = nullptr) {
    if constexpr (sizeof...(T) == 1) {
      using RowType = first_of_variadic_t<T...>;
      const RowSchema* schema = GetRowSchema<RowType>();
      if (schema == nullptr) {
        RUDF_LOG_RETURN_FMT_ERROR("Invalid row object type to get schema");
      }
      int row_idx = GetRowIdx(*schema);
      if (row_idx < 0) {
        RUDF_LOG_RETURN_FMT_ERROR("No row found for schema");
      }
      size_t count = Count();
      if (mask != nullptr) {
        if (mask->Size() != count) {
          RUDF_LOG_RETURN_FMT_ERROR("mask size:{} mismatch table row count:{}", mask->Size(), count);
        }
      }
      if constexpr (std::is_void_v<R>) {
        if (thread_pool != nullptr) {
          thread_pool->Run(0, count, [&](size_t i) {
            if (mask != nullptr) {
              if (!(*mask)[i]) {
                return;
              }
            }
            // RowType* row = rows_[row_idx].GetRowPtrs()[i].As<RowType>();
            RowType* row = rows_[row_idx].GetRowPtrs()->Value(i).template As<RowType>();
            f(static_cast<size_t>(i), row);
          });
          return absl::OkStatus();
        }
      }

      for (int i = 0; i < static_cast<int>(count); i++) {
        if (mask != nullptr) {
          if (!(*mask)[i]) {
            continue;
          }
        }
        // RowType* row = rows_[row_idx].GetRowPtrs()[i].As<RowType>();
        RowType* row = rows_[row_idx].GetRowPtrs()->Value(i).template As<RowType>();
        if constexpr (std::is_void_v<R>) {
          f(static_cast<size_t>(i), row);
        } else {
          R r = f(static_cast<size_t>(i), row);
          auto code = HandleIteratorValue(r);
          switch (code) {
            case VisitStatusCode::kReset: {
              i = -1;
              break;
            }
            case VisitStatusCode::kExit: {
              return absl::OkStatus();
            }
            case VisitStatusCode::kNext:
            default: {
              break;
            }
          }
        }
      }
    } else {
      std::vector<RowSchema> schemas;
      (schemas.emplace_back(NewRowSchema<T>()), ...);
      auto status = Validate(schemas);
      if (!status.ok()) {
        return status;
      }

      size_t count = Count();
      if (mask != nullptr) {
        if (mask->Size() != count) {
          RUDF_LOG_RETURN_FMT_ERROR("mask size:{} mismatch table row count:{}", mask->Size(), count);
        }
      }

      if constexpr (std::is_void_v<R>) {
        if (thread_pool != nullptr) {
          thread_pool->Run(0, count, [&](size_t i) {
            if (mask != nullptr) {
              if (!(*mask)[i]) {
                return;
              }
            }
            VisitRow(static_cast<size_t>(i), f, std::index_sequence_for<T...>{});
          });
          return absl::OkStatus();
        }
      }

      for (int i = 0; i < static_cast<int>(count); i++) {
        if (mask != nullptr) {
          if (!(*mask)[i]) {
            continue;
          }
        }
        if constexpr (std::is_void_v<R>) {
          VisitRow(static_cast<size_t>(i), f, std::index_sequence_for<T...>{});
        } else {
          R r = VisitRow(static_cast<size_t>(i), f, std::index_sequence_for<T...>{});
          auto code = HandleIteratorValue(r);
          switch (code) {
            case VisitStatusCode::kReset: {
              i = -1;
              break;
            }
            case VisitStatusCode::kExit: {
              return absl::OkStatus();
            }
            case VisitStatusCode::kNext:
            default: {
              break;
            }
          }
        }
      }
    }
    return absl::OkStatus();
  }
  template <typename... T>
  Vector<Bit>* Filter(typename VisitorSignatureHelper<bool, T...>::type f,
                      std::shared_ptr<ThreadPool> thread_pool = nullptr) {
    size_t count = Count();
    Vector<Bit>* filter_mask = ctx_.NewVector<Bit>(count);
    // memset(filter_mask->MutableData(), 0, filter_mask.BytesCapacity());
    Foreach<void, T...>(
        [&](size_t idx, const T*... args) {
          if (f(idx, args...)) {
            filter_mask->Set(idx, Bit(true));
          }
        },
        thread_pool);
    return filter_mask;
  }
  Table* Filter(Vector<Bit>* bits);

  Vector<Bit>* Dedup(StringView column, uint32_t k);

  std::pair<Table*, Table*> Split(Vector<Bit>* bits);

  Table* OrderBy(StringView column, bool descending);
  template <typename T>
  Table* OrderBy(Vector<T>* by, bool descending);
  template <typename T>
  Table* Topk(Vector<T>* by, uint32_t k, bool descending);
  Table* Head(uint32_t k);
  Table* Tail(uint32_t k);
  template <typename T>
  absl::Span<Table*> GroupBy(Vector<T>* by);
  absl::Span<Table*> GroupBy(absl::Span<const StringView> columns);
  absl::Span<Table*> GroupBy(StringView column) { return GroupBy(std::vector<StringView>{column}); }

  Table* Concat(Table* other);
  Table* Concat(Table::SmartPtr& ptr) { return Concat(ptr.get()); }

  absl::Status Distinct(absl::Span<const StringView> columns);
  absl::Status Distinct(const std::vector<StringView>& columns) {
    return Distinct(absl::Span<const StringView>(columns));
  }
  absl::Status Distinct(StringView column) {
    std::vector<StringView> columns{column};
    return Distinct(absl::Span<StringView>{columns});
  }
  template <typename... T>
  absl::Status Distinct(absl::Span<const StringView> columns, typename MergeVisitorSignatureHelper<T...>::type merge) {
    std::vector<RowSchema> schemas;
    (schemas.emplace_back(NewRowSchema<T>()), ...);
    auto status = Validate(schemas);
    if (!status.ok()) {
      return status;
    }
    std::vector<std::vector<const uint8_t*>> distinct_objs;
    distinct_objs.resize(rows_.size());
    size_t add_distinct_obj_cursor = 0;
    auto add_distinct_obj = [&](const void* p) {
      distinct_objs[add_distinct_obj_cursor++].emplace_back(reinterpret_cast<const uint8_t*>(p));
    };
    DistinctIndiceTable indice_table = DistinctByColumns(columns);

    for (auto& [indice, duplicate_indices] : indice_table) {
      std::tuple<T*...> exist_row = LoadMutableRow<T...>(indice, std::index_sequence_for<T...>{});
      for (auto duplicate_indice : duplicate_indices) {
        std::tuple<T*...> to_merge_row = LoadMutableRow<T...>(duplicate_indice, std::index_sequence_for<T...>{});
        std::apply(
            [&](auto&... exist_row_element) {
              std::apply(
                  [&](auto&... merge_row_element) {
                    if constexpr (sizeof...(T) == 1) {
                      using RowType = first_of_variadic_t<T...>;
                      RowType* new_row = merge(exist_row_element..., merge_row_element...);
                      exist_row = std::tuple<T*...>(new_row);
                    } else {
                      exist_row = merge(exist_row_element..., merge_row_element...);
                    }
                  },
                  to_merge_row);
            },
            exist_row);
      }
      std::apply([&](auto&... exist_row_element) { (add_distinct_obj(exist_row_element), ...); }, exist_row);
      add_distinct_obj_cursor = 0;
    }
    for (size_t row_idx = 0; row_idx < rows_.size(); row_idx++) {
      rows_[row_idx].Reset(std::move(distinct_objs[row_idx]));
    }
    UnloadAllColumns();
    return absl::OkStatus();
  }
  template <typename... T>
  absl::Status Distinct(StringView by_column, typename MergeVisitorSignatureHelper<T...>::type merge) {
    return Distinct(std::vector<StringView>{by_column}, merge);
  }

  template <typename... T>
  absl::Status Merge(Table* other, absl::Span<const StringView> by_columns,
                     typename MergeVisitorSignatureHelper<T...>::type merge) {
    auto status = DoConcat(other);
    if (!status.ok()) {
      return status;
    }
    return Distinct(by_columns, merge);
  }
  template <typename... T>
  absl::Status Merge(Table* other, StringView by_column, typename MergeVisitorSignatureHelper<T...>::type merge) {
    return Merge(other, std::vector<StringView>{by_column}, merge);
  }

  template <typename R, typename... T>
  absl::StatusOr<SmartPtr> Map(const TableSchema* new_table_schema,
                               typename MapVisitorSignatureHelper<R, T...>::type f) {
    std::vector<RowSchema> schemas;
    (schemas.emplace_back(NewRowSchema<T>()), ...);
    auto status = Validate(schemas);
    if (!status.ok()) {
      return status;
    }
    SmartPtr map_table = NewTableBySchema(new_table_schema);

    std::vector<RowSchema> map_table_schemas;
    if constexpr (is_specialization<R, std::tuple>::value) {
      GetRowSchemaFromTuple<R, 0>(map_table_schemas);
    } else {
      map_table_schemas.emplace_back(NewRowSchema<R>());
    }
    status = map_table->Validate(map_table_schemas);
    if (!status.ok()) {
      return status;
    }
    std::vector<std::vector<const uint8_t*>> map_table_row_objs;
    map_table_row_objs.resize(map_table->rows_.size());
    size_t add_map_table_row_obj_cursor = 0;
    auto add_map_table_row_obj = [&](const void* p) {
      map_table_row_objs[add_map_table_row_obj_cursor++].emplace_back(reinterpret_cast<const uint8_t*>(p));
    };

    size_t row_count = Count();
    for (size_t i = 0; i < row_count; i++) {
      std::tuple<T*...> exist_row = LoadMutableRow<T...>(i, std::index_sequence_for<T...>{});
      auto map_row = std::apply([&](auto&... exist_row_element) { return f(i, exist_row_element...); }, exist_row);
      if constexpr (is_specialization<R, std::tuple>::value) {
        if (!any_nullptr_impl(map_row)) {
          std::apply([&](auto&... exist_row_element) { (add_map_table_row_obj(exist_row_element), ...); }, map_row);
          add_map_table_row_obj_cursor = 0;
        }
      } else {
        if (map_row != nullptr) {
          map_table_row_objs[0].emplace_back(reinterpret_cast<const uint8_t*>(map_row));
        }
      }
    }
    for (size_t row_idx = 0; row_idx < map_table->rows_.size(); row_idx++) {
      map_table->rows_[row_idx].Reset(std::move(map_table_row_objs[row_idx]));
    }
    return map_table;
  }

  template <typename R, typename... T>
  absl::StatusOr<SmartPtr> FlatMap(const TableSchema* new_table_schema,
                                   typename FlatMapVisitorSignatureHelper<R, T...>::type f) {
    std::vector<RowSchema> schemas;
    (schemas.emplace_back(NewRowSchema<T>()), ...);
    auto status = Validate(schemas);
    if (!status.ok()) {
      return status;
    }
    SmartPtr map_table = NewTableBySchema(new_table_schema);

    std::vector<RowSchema> map_table_schemas;
    if constexpr (is_specialization<R, std::tuple>::value) {
      GetRowSchemaFromTuple<R, 0>(map_table_schemas);
    } else {
      map_table_schemas.emplace_back(NewRowSchema<R>());
    }
    status = map_table->Validate(map_table_schemas);
    if (!status.ok()) {
      return status;
    }
    std::vector<std::vector<const uint8_t*>> map_table_row_objs;
    map_table_row_objs.resize(map_table->rows_.size());
    size_t add_map_table_row_obj_cursor = 0;
    auto add_map_table_row_obj = [&](const void* p) {
      map_table_row_objs[add_map_table_row_obj_cursor++].emplace_back(reinterpret_cast<const uint8_t*>(p));
    };

    size_t row_count = Count();
    for (size_t i = 0; i < row_count; i++) {
      std::tuple<T*...> exist_row = LoadMutableRow<T...>(i, std::index_sequence_for<T...>{});
      auto flatmap_rows = std::apply([&](auto&... exist_row_element) { return f(i, exist_row_element...); }, exist_row);
      if constexpr (is_specialization<R, std::tuple>::value) {
        for (auto& flatmap_row : flatmap_rows) {
          if (!any_nullptr_impl(flatmap_row)) {
            std::apply([&](auto&... exist_row_element) { (add_map_table_row_obj(exist_row_element), ...); },
                       flatmap_row);
            add_map_table_row_obj_cursor = 0;
          }
        }
      } else {
        for (auto p : flatmap_rows) {
          if (p != nullptr) {
            map_table_row_objs[0].emplace_back(reinterpret_cast<const uint8_t*>(p));
          }
        }
      }
    }
    for (size_t row_idx = 0; row_idx < map_table->rows_.size(); row_idx++) {
      map_table->rows_[row_idx].Reset(std::move(map_table_row_objs[row_idx]));
    }
    return map_table;
  }

  /**
  ** return column count
  */
  size_t Size() const;
  /**
   ** return row count
   */
  size_t Count() const;
  std::array<size_t, 2> Shape() const;

  Context& GetContext() { return ctx_; }

  const TableSchema* GetTableSchema() const { return reinterpret_cast<const TableSchema*>(schema_); }

  template <class R, class... Args>
  absl::StatusOr<R> EvalFunction(const std::string& source, Args... args) {
    return exec::eval_function<R, Args...>(source, args...);
  }

  template <class R, class... Args>
  absl::StatusOr<R> EvalExpression(const std::string& source, const std::vector<std::string>& arg_names, Args... args) {
    return exec::eval_expression<R, Args...>(source, arg_names, args...);
  }

 private:
  using DistinctKey = std::vector<std::pair<const DType*, const uint8_t*>>;
  struct DistinctKeyHash {
    std::size_t operator()(const DistinctKey& ks) const {
      size_t h = ks[0].first->Hash(ks[0].second);
      for (size_t i = 1; i < ks.size(); i++) {
        h = h ^ (ks[i].first->Hash(ks[i].second));
      }
      return h;
    }
  };
  struct DistinctKeyCompare {
    bool operator()(const DistinctKey& left, const DistinctKey& right) const {
      for (size_t i = 0; i < left.size(); i++) {
        if (!left[i].first->Equal(left[i].second, right[i].second)) {
          return false;
        }
      }
      return true;
    }
  };

  Table(Context& ctx, const DynObjectSchema* s);
  Table(Table&);
  Table* NewTableBySchema(const std::string& schema);
  SmartPtr NewTableBySchema(const TableSchema* schema);
  Table* Clone();

  std::vector<int32_t> GetIndices();
  void SetIndices(std::vector<int32_t>&& indices);

  absl::Status DoAddRows(std::vector<PartialRows>&& rows);
  absl::Status InsertRow(size_t pos, const std::vector<PartialRow>& row);

  absl::StatusOr<uint32_t> GetColumnOffset(const std::string& name);

  bool IsColumnLoaded(uint32_t offset);

  size_t GetColumnMemorySize(const DType& dtype);

  template <typename T>
  absl::StatusOr<Vector<T>*> GetColumn(const std::string& name) {
    uint32_t offset = 0;
    auto result = DynObject::Get<Vector<T>*>(name, &offset);
    if (!result.ok()) {
      return result;
    }
    Vector<T>* vec = std::move(result.value());
    if (vec == nullptr) {
      vec = GetColumnByOffset<T>(offset);
    }
    return vec;
  }
  // VectorBuf GetColumnVectorBuf(StringView name);
  template <typename T>
  RowSchema NewRowSchema() {
    if constexpr (std::is_base_of_v<::google::protobuf::Message, T>) {
      static T msg;
      const ::google::protobuf::Descriptor* desc = msg.GetDescriptor();
      return RowSchema(desc);
    } else if constexpr (std::is_base_of_v<::flatbuffers::Table, T>) {
      auto* type_table = T::MiniReflectTypeTable();
      return RowSchema(type_table);
    } else {
      return RowSchema(get_dtype<T>());
    }
  }

  template <typename T>
  const RowSchema* GetRowSchema() {
    return GetRowSchema(NewRowSchema<T>());
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
  absl::Span<Table*> GroupBy(const T* by, size_t n);

  Table* GatherSubTable(std::vector<int32_t>& indices);

  void SetColumn(uint32_t offset, VectorBase* vec);

  absl::StatusOr<VectorBase*> GatherField(uint8_t* vec_ptr, const DType& dtype, uint32_t offset,
                                          Vector<int32_t> indices);

  template <typename T>
  absl::Status LoadProtobufColumn(Vector<Pointer>* pb_vector, const Column& column);
  template <typename T>
  absl::Status LoadFlatbuffersColumn(Vector<Pointer>* fbs_vector, const Column& column);
  template <typename T>
  absl::Status LoadStructColumn(Vector<Pointer>* struct_vector, const Column& column);

  template <typename T>
  absl::Status LoadColumn(Vector<Pointer>* objs, const Column& column);

  absl::Status LoadColumn(Rows& rows, const Column& column);
  const RowSchema* GetRowSchema(const RowSchema& schema);
  int GetRowIdx(const RowSchema& schema);
  absl::Status Validate(const std::vector<RowSchema>& schemas);

  template <typename T>
  Vector<Bit>* Dedup(Vector<T>* data, size_t n, size_t k);

  template <typename T>
  absl::Status Set(const std::string& name, T&& v) {
    return DynObject::Set(name, std::forward<T>(v));
  }

  template <typename R>
  VisitStatusCode HandleIteratorValue(R v) {
    if constexpr (std::is_same_v<VisitStatusCode, R>) {
      return v;
    } else if constexpr (std::is_integral_v<R>) {
      if (v == 0) {
        return VisitStatusCode::kNext;
      } else {
        return VisitStatusCode::kExit;
      }
    } else {
      return VisitStatusCode::kNext;
    }
  }

  DistinctIndiceTable DistinctByColumns(absl::Span<const StringView> columns);
  void DoFilter(Vector<Bit>* bits);

  absl::Status DoConcat(Table* other);

  template <typename TupleLike, std::size_t I = 0>
  void GetRowSchemaFromTuple(std::vector<RowSchema>& schemas) {
    if constexpr (I < std::tuple_size_v<TupleLike>) {
      using SelectedType = std::tuple_element_t<I, TupleLike>;
      schemas.emplace_back(NewRowSchema<SelectedType>());
      GetRowSchemaFromTuple<TupleLike, I + 1>();
    }
  }

  template <typename T>
  const T* LoadRowElement(size_t i, size_t j) {
    // const T* row = rows_[j].GetRowPtrs()[i].As<T>();
    const T* row = rows_[j].GetRowPtrs()->Value(i).template As<T>();
    return row;
  }
  template <typename T>
  T* LoadMutableRowElement(size_t i, size_t j) {
    // T* row = rows_[j].GetRowPtrs()[i].As<T>();
    T* row = rows_[j].GetRowPtrs()->Value(i).template As<T>();
    return row;
  }
  template <typename... T, size_t... Is>
  std::tuple<T*...> LoadMutableRow(size_t row_idx, std::index_sequence<Is...>) {
    return std::make_tuple(LoadMutableRowElement<T>(row_idx, Is)...);
  }

  template <typename R, typename... T, size_t... Is>
  R VisitRow(size_t row_idx, std::function<R(size_t, const T*...)>& f, std::index_sequence<Is...>) {
    return f(row_idx, LoadRowElement<T>(row_idx, Is)...);
  }

  template <typename R, typename... T, size_t... Is>
  R MergeRow(size_t current_row_idx, size_t merge_row_idx, std::function<R(T*..., const T*...)>& f,
             std::index_sequence<Is...>) {
    return f(LoadMutableRowElement<T>(current_row_idx, Is)..., LoadRowElement<T>(merge_row_idx, Is)...);
  }

  Context& ctx_;
  std::vector<int32_t> indices_;
  std::vector<Rows> rows_;
  friend class TableSchema;
};
}  // namespace table
}  // namespace rapidudf