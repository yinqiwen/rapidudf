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

#include "rapidudf/types/simd/table.h"
namespace rapidudf {
namespace simd {

/**
 ** defined in rapidudf/builtin/simd_vector/ops.h
 ** defined in rapidudf/builtin/simd_vector/column_ops.h
 */
template <typename T>
Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n);

Column* simd_column_filter(Column* data, Column* bits);
Column* simd_column_gather(Column* data, Column* indices);

/**
 ** defined in rapidudf/builtin/simd_vector/table_ops.h
 */
Table* simd_table_filter(simd::Table* table, simd::Column* bits);
Table* simd_table_order_by(simd::Table* table, simd::Column* by, bool descending);
Table* simd_table_topk(simd::Table* table, simd::Column* by, uint32_t k, bool descending);
Table* simd_table_take(simd::Table* table, uint32_t k);

void Table::Set(const std::string& name, Column* column) { column_table_[name] = column; }

absl::Status Table::Add(const std::string& name, Column* column) {
  std::string column_name = name;
  if (column_name.empty()) {
    column_name = fmt::format("__column_{}", column_table_.size());
  }
  if (column_table_.find(column_name) != column_table_.end()) {
    return absl::InvalidArgumentError(fmt::format("duplicate column name:{} to add", column_name));
  }
  column_table_.emplace(column_name, column);
  return absl::OkStatus();
}

absl::StatusOr<Column**> Table::Get(StringView name) {
  auto found = column_table_.find(std::string_view(name));
  if (found == column_table_.end()) {
    return absl::NotFoundError("");
  }
  return &found->second;
}

Column* Table::operator[](StringView name) {
  auto result = Get(name);
  if (result.ok()) {
    return *(result.value());
  }
  throw std::logic_error(fmt::format("No column:{} found in table.", name));
}
Table* Table::Filter(Column* bits) { return simd_table_filter(this, bits); }
Table* Table::OrderBy(simd::Table* table, simd::Column* by, bool descending) {
  return simd_table_order_by(this, by, descending);
}
Table* Table::Topk(simd::Table* table, simd::Column* by, uint32_t k, bool descending) {
  return simd_table_topk(this, by, k, descending);
}
Table* Table::Take(uint32_t k) { return simd_table_take(this, k); }

void Table::Visit(std::function<void(const std::string&, Column*)>&& f) {
  for (auto& [name, column] : column_table_) {
    f(name, column);
  }
}

size_t Table::Size() const { return column_table_.size(); }

Vector<int32_t> Table::GetIndices() {
  if (column_table_.empty()) {
    throw std::logic_error(fmt::format("Can NOT get indices from empty table"));
  }
  if (indices_.Size() == 0) {
    auto* first_column = (column_table_.begin()->second);
    indices_ = simd_vector_iota<int32_t>(ctx_, 0, first_column->size());
  }
  auto* p = ctx_.ArenaAllocate(sizeof(int32_t) * indices_.Size());
  memcpy(p, indices_.Data(), sizeof(int32_t) * indices_.Size());
  VectorData vdata(p, indices_.Size(), sizeof(int32_t) * indices_.Size());
  vdata.SetTemporary(true);
  return Vector<int32_t>(vdata);
}
}  // namespace simd
}  // namespace rapidudf