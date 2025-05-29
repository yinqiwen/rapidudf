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
#include <stdexcept>
#include <tuple>

#include "rapidudf/functions/names.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/reflect/struct.h"
#include "rapidudf/table/table.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"
namespace rapidudf {
namespace functions {
struct SimdTableHelper {
  static size_t column_count(table::Table* table) { return table->Size(); }

  /**
  ** filter table by bits
  */
  static table::Table* filter(table::Table* table, Vector<Bit> bits) { return table->Filter(bits); }
  /**
  **   Sort table by given column
  */
  template <typename T>
  static table::Table* order_by(table::Table* table, Vector<T> by, bool descending) {
    return table->OrderBy(by, descending);
  }
  static table::Table* order_by_column(table::Table* table, StringView by, bool descending) {
    return table->OrderBy(by, descending);
  }

  /**
  **   Sort & Returns the first k rows as a list of Row.
  */
  template <typename T>
  static table::Table* topk(table::Table* table, Vector<T> by, uint32_t k, bool descending) {
    return table->Topk(by, k, descending);
  }
  /**
  **   Returns the first num rows as a list of Row.
  */
  static table::Table* head(table::Table* table, uint32_t k) { return table->Head(k); }

  /**
   **   Returns the first num rows as a list of Row.
   */
  static table::Table* tail(table::Table* table, uint32_t k) { return table->Tail(k); }

  /**
   **   Returns table row count
   */
  static uint32_t count(table::Table* table) { return table->Count(); }

  /**
   **   Returns the first num rows as a list of Row.
   */
  static table::Table* concat(table::Table* table, table::Table* other) { return table->Concat(other); }

  template <typename T>
  static absl::Span<table::Table*> group_by(table::Table* table, Vector<T> by) {
    return table->GroupBy(by);
  }
  static absl::Span<table::Table*> group_by_column(table::Table* table, StringView by) { return table->GroupBy(by); }

  template <typename T>
  static Vector<T> get_column(table::Table* table, uint32_t offset) {
    // return Vector<T>(table->GetColumnByOffset(offset));
    return table->LoadColumnByOffset<T>(offset);
  }

  static void Init() {
    RUDF_STRUCT_HELPER_METHODS_BIND(SimdTableHelper, column_count, filter, head, tail, count, concat);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_f32", topk<float>);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_f64", topk<double>);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_u32", topk<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_i32", topk<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_u64", topk<uint64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("topk_i64", topk<int64_t>);

    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_f32", order_by<float>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_f64", order_by<double>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_u32", order_by<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_i32", order_by<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_u64", order_by<uint64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by_i64", order_by<int64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("order_by", order_by_column);

    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_f32", group_by<float>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_f64", group_by<double>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_u8", group_by<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_i8", group_by<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_u16", group_by<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_i26", group_by<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_u32", group_by<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_i32", group_by<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_u64", group_by<uint64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_i64", group_by<int64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND("group_by_string_view", group_by<StringView>);

    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_F32), get_column<float>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_F64), get_column<double>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_U8), get_column<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_I8), get_column<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_U16), get_column<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_I16), get_column<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_U32), get_column<uint32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_I32), get_column<int32_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_U64), get_column<uint64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_I64), get_column<int64_t>);
    RUDF_STRUCT_HELPER_METHOD_BIND(GetFunctionName(kTableGetColumnFunc, DATA_STRING_VIEW), get_column<StringView>);
  }
};
void init_builtin_simd_table_funcs() { SimdTableHelper::Init(); }
}  // namespace functions
}  // namespace rapidudf