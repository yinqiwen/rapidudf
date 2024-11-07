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
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/reflect/reflect.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/column.h"
#include "rapidudf/vector/row.h"
#include "rapidudf/vector/table.h"
#include "rapidudf/vector/vector.h"

namespace rapidudf {

namespace simd {

struct TableCreateOptions {
  std::unordered_set<std::string> includes;
  std::unordered_set<std::string> excludes;
  bool ignore_unsupported_fields = false;
  bool IsAllowed(const std::string& field) const;
};

class TableSchema : public DynObjectSchema {
 public:
  using InitFunc = std::function<void(TableSchema* s)>;
  static const TableSchema* GetOrCreate(const std::string& name, InitFunc&& init, const TableCreateOptions& opts = {});
  static const TableSchema* Get(const std::string& name);
  static TableSchema* GetMutable(const std::string& name);

  typename Table::SmartPtr NewTable(Context& ctx) const;

  template <typename T>
  absl::Status AddColumns() {
    if constexpr (std::is_base_of_v<::google::protobuf::Message, T>) {
      static T msg;
      return AddColumns(&msg);
    } else if constexpr (std::is_base_of_v<::flatbuffers::Table, T>) {
      return AddColumns(T::MiniReflectTypeTable());
    } else {
      return AddColumns(get_dtype<T>());
    }
  }

  template <typename T>
  absl::Status AddColumn(const std::string& name) {
    return AddColumn<T>(name, nullptr, 0);
  }
  bool ExistColumn(const std::string& name, const DType& dtype) const;
  bool ExistRow(const RowSchema& row) const;

  const reflect::Column* GetColumnByIdx(uint32_t idx) const;

 private:
  TableSchema(const std::string& name, Options opts, const TableCreateOptions& table_opts)
      : DynObjectSchema(name, opts), table_opts_(table_opts) {}
  template <typename T>
  absl::Status AddField(const std::string& name) {
    return absl::UnimplementedError("AddField");
  }

  template <typename T>
  absl::Status AddColumn(const std::string& name, const RowSchema* schema, uint32_t field_idx) {
    if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      return AddColumn(name, get_dtype<simd::Vector<StringView>>(), schema, field_idx);
    } else if constexpr (std::is_same_v<bool, T>) {
      return AddColumn(name, get_dtype<simd::Vector<Bit>>(), schema, field_idx);
    } else {
      return AddColumn(name, get_dtype<simd::Vector<T>>(), schema, field_idx);
    }
  }

  typename DynObject::SmartPtr NewObject() const { return nullptr; }

  absl::Status AddColumns(const ::google::protobuf::Message* msg);
  absl::Status AddColumns(const flatbuffers::TypeTable* type_table);
  absl::Status AddColumns(const DType& dtype);
  absl::Status AddColumn(const std::string& name, const DType& dtype, const RowSchema* schema, uint32_t field_idx);

  TableCreateOptions table_opts_;
  std::vector<RowSchemaPtr> row_schemas_;
  std::vector<reflect::Column> columns_;

  friend class Table;
};
}  // namespace simd
}  // namespace rapidudf