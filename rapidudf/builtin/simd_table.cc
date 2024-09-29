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
#include "rapidudf/builtin/simd_vector/column_ops.h"
#include "rapidudf/builtin/simd_vector/ops.h"
#include "rapidudf/builtin/simd_vector/table_ops.h"
#include "rapidudf/reflect/struct.h"
#include "rapidudf/types/simd_vector_table.h"
namespace rapidudf {
struct SimdTableHelper {
  static simd::Column** get(simd::Table* table, StringView name) {
    auto result = table->Get(name);
    if (!result.ok()) {
      throw std::logic_error(result.status().ToString());
    }
    return (result.value());
  }
  static void add(simd::Table* table, StringView name, simd::Column* column) {
    auto status = table->Add(name.str(), column);
    if (!status.ok()) {
      throw std::logic_error(status.ToString());
    }
  }
  static void set(simd::Table* table, StringView name, simd::Column* column) { table->Set(name.str(), column); }
  static size_t size(simd::Table* table) { return table->Size(); }

  /**
  ** filter table by bits
  */
  static simd::Table* filter(simd::Table* table, simd::Column* bits) { return simd::simd_table_filter(table, bits); }
  /**
  **   Sort table by given column
  */
  static simd::Table* order_by(simd::Table* table, simd::Column* by, bool descending) {
    return simd::simd_table_order_by(table, by, descending);
  }
  /**
  **   Sort & Returns the first k rows as a list of Row.
  */
  static simd::Table* topk(simd::Table* table, simd::Column* by, uint32_t k, bool descending) {
    return simd::simd_table_topk(table, by, k, descending);
  }
  /**
  **   Returns the first num rows as a list of Row.
  */
  static simd::Table* take(simd::Table* table, uint32_t k) { return simd::simd_table_take(table, k); }

  static void Init() {
    RUDF_STRUCT_HELPER_METHODS_BIND(SimdTableHelper, get, add, set, size, filter, order_by, topk, take)
  }
};
void init_builtin_simd_table_funcs() { SimdTableHelper::Init(); }

}  // namespace rapidudf