/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/builtin/simd_vector/table_ops.h"
#include "rapidudf/builtin/simd_vector/column_ops.h"
namespace rapidudf {
namespace simd {
Table* simd_table_filter(simd::Table* table, simd::Column* bits) {
  auto& ctx = table->GetContext();
  simd::Table* new_table = ctx.New<simd::Table>(ctx);
  table->Visit([&](const std::string& name, simd::Column* c) {
    simd::Column* new_column = simd::simd_column_filter(c, bits);
    std::ignore = new_table->Add(name, new_column);
  });
  return new_table;
}

simd::Table* simd_table_take(simd::Table* table, uint32_t k) {
  auto& ctx = table->GetContext();
  simd::Table* new_table = ctx.New<simd::Table>(ctx);
  table->Visit([&](const std::string& name, simd::Column* c) {
    auto* new_column = c->take(k);
    std::ignore = new_table->Add(name, new_column);
  });
  return new_table;
}

simd::Table* simd_table_topk(simd::Table* table, simd::Column* by, uint32_t k, bool descending) {
  auto& ctx = table->GetContext();
  simd::Table* new_table = ctx.New<simd::Table>(ctx);
  simd::Column* indices = simd::Column::FromVector(ctx, table->GetIndices());
  simd::column_topk_key_value(by, indices, k, descending);
  indices = indices->take(k);
  table->Visit([&](const std::string& name, simd::Column* c) {
    simd::Column* new_column = nullptr;
    if (c != by) {
      new_column = simd::simd_column_gather(c, indices);
    } else {
      new_column = by->take(k);
    }
    std::ignore = new_table->Add(name, new_column);
  });
  return new_table;
}

simd::Table* simd_table_order_by(simd::Table* table, simd::Column* by, bool descending) {
  auto& ctx = table->GetContext();
  simd::Table* new_table = ctx.New<simd::Table>(ctx);
  simd::Column* indices = simd::Column::FromVector(ctx, table->GetIndices());

  simd::column_sort_key_value(by, indices, descending);
  table->Visit([&](const std::string& name, simd::Column* c) {
    simd::Column* new_column = nullptr;
    if (c != by) {
      new_column = simd::simd_column_gather(c, indices);
    } else {
      new_column = by;
    }
    std::ignore = new_table->Add(name, new_column);
  });
  return new_table;
}
}  // namespace simd
}  // namespace rapidudf