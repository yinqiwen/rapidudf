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
#include "rapidudf/types/simd_vector_table.h"
#include <fmt/format.h>
#include <memory>
#include <variant>
namespace rapidudf {
namespace simd {

size_t Column::Size() const {
  return std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          return size_t(0);
        } else {
          return arg.Size();
        }
      },
      data_);
}

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

// absl::Status Table::AddColumn(const std::string& name, Column column) {
//   std::string column_name = name;
//   if (column_name.empty()) {
//     column_name = fmt::format("__column_{}", column_table_.size());
//   }
//   if (column_table_.find(column_name) != column_table_.end()) {
//     return absl::InvalidArgumentError(fmt::format("duplicate column name:{} to add", column_name));
//   }
//   auto p = std::make_unique<Column>(std::move(column));
//   column_table_.emplace(column_name, p.get());
//   columns_.emplace_back(std::move(p));
//   return absl::OkStatus();
// }

absl::StatusOr<Column**> Table::Get(StringView name) {
  auto found = column_table_.find(std::string_view(name));
  if (found == column_table_.end()) {
    return absl::NotFoundError("");
  }
  return &found->second;
}
size_t Table::Size() const { return column_table_.size(); }
}  // namespace simd
}  // namespace rapidudf