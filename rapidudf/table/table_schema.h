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
#include <unordered_set>
#include <vector>
#include "google/protobuf/message.h"

#include "rapidudf/context/context.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/table/column.h"
#include "rapidudf/table/row.h"
#include "rapidudf/table/table.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {
namespace table {
struct TableColumnOptions {
  std::unordered_set<std::string> include_fields;
  std::unordered_set<std::string> exclude_fields;
  std::string prefix;
  bool ignore_unsupported_fields = false;
  bool IsAllowed(const std::string& field) const;
};

class TableSchema : public DynObjectSchema {
 public:
  using InitFunc = std::function<void(TableSchema* s)>;
  static const TableSchema* GetOrCreate(const std::string& name, InitFunc&& init);
  static const TableSchema* Get(const std::string& name);
  static TableSchema* GetMutable(const std::string& name);

  typename Table::SmartPtr NewTable(Context& ctx) const;

  template <typename T>
  absl::Status AddColumns(const TableColumnOptions& opts = {}) {
    if constexpr (std::is_base_of_v<::google::protobuf::Message, T>) {
      static T msg;
      return AddColumns(opts, &msg);
    } else if constexpr (std::is_base_of_v<::flatbuffers::Table, T>) {
      return AddColumns(opts, T::MiniReflectTypeTable());
    } else {
      return AddColumns(opts, get_dtype<T>());
    }
  }

  bool ExistColumn(const std::string& name, const DType& dtype) const;
  bool ExistRow(const RowSchema& row) const;

  const Column* GetColumnByIdx(uint32_t idx) const;

 private:
  TableSchema(const std::string& name, Options opts) : DynObjectSchema(name, opts) {}
  template <typename T>
  absl::Status AddField(const std::string& name) {
    return absl::UnimplementedError("AddField");
  }

  template <typename T>
  absl::Status AddColumn(const std::string& name, const RowSchema* schema, uint32_t field_idx) {
    if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      return AddColumn(name, get_dtype<Vector<StringView>>(), schema, field_idx);
    } else if constexpr (std::is_same_v<bool, T>) {
      return AddColumn(name, get_dtype<Vector<Bit>>(), schema, field_idx);
    } else {
      return AddColumn(name, get_dtype<Vector<T>>(), schema, field_idx);
    }
  }

  typename DynObject::SmartPtr NewObject() const { return nullptr; }

  absl::Status AddColumns(const TableColumnOptions& opts, const ::google::protobuf::Message* msg);
  absl::Status AddColumns(const TableColumnOptions& opts, const flatbuffers::TypeTable* type_table);
  absl::Status AddColumns(const TableColumnOptions& opts, const DType& dtype);
  absl::Status AddColumn(const std::string& name, const DType& dtype, const RowSchema* schema, uint32_t field_idx);

  std::vector<RowSchemaPtr> row_schemas_;
  std::vector<Column> columns_;

  friend class Table;
};
}  // namespace table
}  // namespace rapidudf