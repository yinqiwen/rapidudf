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
#include <stdexcept>
#include <tuple>

#include "rapidudf/reflect/struct.h"
#include "rapidudf/types/simd/table.h"
#include "rapidudf/types/simd/vector.h"
namespace rapidudf {
namespace functions {
struct SimdTableHelper {
  //   static simd::Column** get(simd::Table* table, StringView name) {
  //     auto result = table->Get(name);
  //     if (!result.ok()) {
  //       throw std::logic_error(result.status().ToString());
  //     }
  //     return (result.value());
  //   }
  //   static void add(simd::Table* table, StringView name, simd::Column* column) {
  //     auto status = table->Add(name.str(), column);
  //     if (!status.ok()) {
  //       throw std::logic_error(status.ToString());
  //     }
  //   }
  //   static void set(simd::Table* table, StringView name, simd::Column* column) { table->Set(name.str(), column); }
  static size_t size(simd::Table* table) { return table->Size(); }

  /**
  ** filter table by bits
  */
  static simd::Table* filter(simd::Table* table, simd::Vector<Bit> bits) { return table->Filter(bits); }
  /**
  **   Sort table by given column
  */
  template <typename T>
  static simd::Table* order_by(simd::Table* table, simd::Vector<T> by, bool descending) {
    return table->OrderBy(by, descending);
  }
  /**
  **   Sort & Returns the first k rows as a list of Row.
  */
  template <typename T>
  static simd::Table* topk(simd::Table* table, simd::Vector<T> by, uint32_t k, bool descending) {
    return table->Topk(by, k, descending);
  }
  /**
  **   Returns the first num rows as a list of Row.
  */
  static simd::Table* take(simd::Table* table, uint32_t k) { return table->Take(k); }

  static void Init() {
    RUDF_STRUCT_HELPER_METHODS_BIND(SimdTableHelper, size, filter, take);
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
  }
};
void init_builtin_simd_table_funcs() { SimdTableHelper::Init(); }
}  // namespace functions
}  // namespace rapidudf