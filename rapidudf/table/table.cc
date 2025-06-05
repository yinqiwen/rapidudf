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

#include "rapidudf/table/table.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "flatbuffers/minireflect.h"

#include "rapidudf/common/allign.h"
#include "rapidudf/functions/simd/bits.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/table/column.h"
#include "rapidudf/table/row.h"
#include "rapidudf/table/table_schema.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/dyn_object_impl.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {
namespace table {
static std::vector<int32_t> get_indices(size_t n) {
  static constexpr uint32_t kDefaultIndiceCount = 10000;
  static std::vector<int32_t> default_indices;
  if (default_indices.empty()) {
    default_indices.resize(kDefaultIndiceCount);
    std::iota(default_indices.begin(), default_indices.end(), 0);
  }
  std::vector<int32_t> indices(n);
  if (n <= kDefaultIndiceCount) {
    memcpy(&indices[0], &default_indices[0], sizeof(int32_t) * n);
  } else {
    std::iota(indices.begin(), indices.end(), 0);
  }
  return indices;
}

Table::Table(Context& ctx, const DynObjectSchema* s) : DynObject(s), ctx_(ctx) {
  for (auto& s : GetTableSchema()->row_schemas_) {
    rows_.emplace_back(Rows(ctx, {}, *s));
  }
  for (size_t i = 0; i < GetTableSchema()->GetColumnCount(); i++) {
    auto column_field = GetTableSchema()->GetColumnByIdx(i);
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + column_field->field.bytes_offset;
    VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
    columns_.emplace_back(std::make_unique<Column>(column_field, vdata));
  }
}
Table::Table(Table& other) : DynObject(other), ctx_(other.ctx_) {
  indices_ = other.indices_;
  for (auto& row : other.rows_) {
    auto ptrs = row.GetRawRowPtrs();
    rows_.emplace_back(Rows(ctx_, std::move(ptrs), row.GetSchema()));
  }
  for (size_t i = 0; i < GetTableSchema()->GetColumnCount(); i++) {
    auto column_field = GetTableSchema()->GetColumnByIdx(i);
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + column_field->field.bytes_offset;
    VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
    columns_.emplace_back(std::make_unique<Column>(column_field, vdata));
  }
}

void Table::Deleter::operator()(Table* ptr) const {
  ptr->~Table();
  uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
  delete[] bytes;
}

std::array<size_t, 2> Table::Shape() const { return {Count(), GetTableSchema()->FieldCount()}; }

uint32_t Table::GetIdxByOffset(uint32_t offset) {
  size_t header_size = align_to<size_t>(sizeof(Table), 16);
  uint32_t idx = (offset - header_size) / sizeof(VectorBase);
  return idx;
}
absl::StatusOr<uint32_t> Table::GetColumnOffset(const std::string& name) {
  auto result = schema_->GetField(name);
  if (!result.ok()) {
    return result.status();
  }
  auto [_, field_offset] = result.value();
  return field_offset;
}

void Table::SetColumnSize(const Column& column, const void* p) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + column.GetFieldOffset();
  VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  if (vdata->Data<void>() == p) {
    vdata->SetSize(Count());
  }
  vdata->SetWritable();
}

absl::Status Table::LoadColumn(const Rows& rows, Column& column) {
  auto objs = rows.GetRowPtrs();
  if (column.GetFieldDType().Elem().IsPrimitive()) {
    uint8_t* column_mem = column.ReserveMemory(Count());
    for (size_t i = 0; i < objs.Size(); i++) {
      const uint8_t* obj = objs[i].As<uint8_t>();
      switch (column.GetFieldDType().Elem().GetFundamentalType()) {
        case DATA_F32: {
          reinterpret_cast<float*>(column_mem)[i] = column.GetF32(obj);
          break;
        }
        case DATA_F64: {
          reinterpret_cast<double*>(column_mem)[i] = column.GetF64(obj);
          break;
        }
        case DATA_U64: {
          reinterpret_cast<uint64_t*>(column_mem)[i] = column.GetU64(obj);
          break;
        }
        case DATA_U32: {
          reinterpret_cast<uint32_t*>(column_mem)[i] = column.GetU32(obj);
          break;
        }
        case DATA_U16: {
          reinterpret_cast<uint16_t*>(column_mem)[i] = column.GetU16(obj);
          break;
        }
        case DATA_U8: {
          reinterpret_cast<uint8_t*>(column_mem)[i] = column.GetU8(obj);
          break;
        }
        case DATA_I64: {
          reinterpret_cast<int64_t*>(column_mem)[i] = column.GetI64(obj);
          break;
        }
        case DATA_I32: {
          reinterpret_cast<int32_t*>(column_mem)[i] = column.GetI32(obj);
          break;
        }
        case DATA_I16: {
          reinterpret_cast<int16_t*>(column_mem)[i] = column.GetI16(obj);
          break;
        }
        case DATA_I8: {
          reinterpret_cast<int8_t*>(column_mem)[i] = column.GetI8(obj);
          break;
        }
        case DATA_STRING_VIEW: {
          reinterpret_cast<StringView*>(column_mem)[i] = column.GetString(obj);
          break;
        }
        case DATA_BIT: {
          bits_set(reinterpret_cast<uint8_t*>(column_mem), i, column.GetBool(obj));
          break;
        }
        default: {
          RUDF_LOG_RETURN_FMT_ERROR("Unsupported column:{} with dtype:{}", column.Name(), column.GetFieldDType());
        }
      }
    }
    SetColumnSize(column, column_mem);
  } else if (column.GetFieldDType().Elem().IsAbslSpan()) {
    uint8_t* column_mem = column.ReserveMemory(Count());
    for (size_t i = 0; i < objs.Size(); i++) {
      const uint8_t* obj = objs[i].As<uint8_t>();
      switch (column.GetFieldDType().Elem().GetFundamentalType()) {
        case DATA_F32: {
          column.GetRepeatedF32(obj, reinterpret_cast<absl::Span<const float>*>(column_mem) + i);
          break;
        }
        case DATA_F64: {
          column.GetRepeatedF64(obj, reinterpret_cast<absl::Span<const double>*>(column_mem) + i);
          break;
        }
        case DATA_U64: {
          column.GetRepeatedU64(obj, reinterpret_cast<absl::Span<const uint64_t>*>(column_mem) + i);
          break;
        }
        case DATA_U32: {
          column.GetRepeatedU32(obj, reinterpret_cast<absl::Span<const uint32_t>*>(column_mem) + i);
          break;
        }
        case DATA_U16: {
          column.GetRepeatedU16(obj, reinterpret_cast<absl::Span<const uint16_t>*>(column_mem) + i);
          break;
        }
        case DATA_U8: {
          column.GetRepeatedU8(obj, reinterpret_cast<absl::Span<const uint8_t>*>(column_mem) + i);
          break;
        }
        case DATA_I64: {
          column.GetRepeatedI64(obj, reinterpret_cast<absl::Span<const int64_t>*>(column_mem) + i);
          break;
        }
        case DATA_I32: {
          column.GetRepeatedI32(obj, reinterpret_cast<absl::Span<const int32_t>*>(column_mem) + i);
          break;
        }
        case DATA_I16: {
          column.GetRepeatedI16(obj, reinterpret_cast<absl::Span<const int16_t>*>(column_mem) + i);
          break;
        }
        case DATA_I8: {
          column.GetRepeatedI8(obj, reinterpret_cast<absl::Span<const int8_t>*>(column_mem) + i);
          break;
        }
        case DATA_STRING_VIEW: {
          column.GetRepeatedString(ctx_, obj, reinterpret_cast<absl::Span<const StringView>*>(column_mem) + i);
          break;
        }
        case DATA_BIT: {
          column.GetRepeatedBool(obj, reinterpret_cast<absl::Span<const bool>*>(column_mem) + i);
          break;
        }
        default: {
          RUDF_LOG_RETURN_FMT_ERROR("Unsupported column:{} with dtype:{}", column.Name(), column.GetFieldDType());
        }
      }
    }
    SetColumnSize(column, column_mem);
    return absl::OkStatus();
  } else {
    RUDF_LOG_RETURN_FMT_ERROR("Unsupported column:{} with column dtype:{}", column.Name(),
                              column.GetFieldDType().Elem());
  }

  return absl::OkStatus();
}

Table* Table::Clone() {
  uint8_t* bytes = new uint8_t[schema_->ByteSize()];
  memset(bytes, 0, schema_->ByteSize());
  try {
    new (bytes) Table(*this);
  } catch (...) {
    throw;
  }
  Table* t = reinterpret_cast<Table*>(bytes);
  Deleter d;
  ctx_.Own(t, d);
  t->ClearAllColumns();
  return t;
}
Table* Table::NewTableBySchema(const std::string& name) {
  auto* table_schema = TableSchema::Get(name);
  if (table_schema == nullptr) {
    return nullptr;
  }
  uint8_t* bytes = new uint8_t[table_schema->ByteSize()];
  memset(bytes, 0, table_schema->ByteSize());
  try {
    new (bytes) Table(ctx_, table_schema);
  } catch (...) {
    throw;
  }
  Table* t = reinterpret_cast<Table*>(bytes);
  Deleter d;
  ctx_.Own(t, d);
  return t;
}
typename Table::SmartPtr Table::NewTableBySchema(const TableSchema* schema) { return schema->NewTable(ctx_); }

std::vector<int32_t> Table::GetIndices() {
  size_t count = Count();
  if (indices_.size() < count) {
    indices_ = get_indices(count);
  } else if (indices_.size() > count) {
    indices_.resize(count);
  }
  return indices_;
}
void Table::SetIndices(std::vector<int32_t>&& indices) { indices_ = std::move(indices); }

absl::Status Table::Validate(const std::vector<RowSchema>& schemas) {
  if (schemas.size() != GetTableSchema()->row_schemas_.size()) {
    RUDF_RETURN_FMT_ERROR("Expected {} column set, but {} given", GetTableSchema()->row_schemas_.size(),
                          schemas.size());
  }
  for (size_t i = 0; i < schemas.size(); i++) {
    if (rows_[i].GetSchema() != schemas[i]) {
      RUDF_RETURN_FMT_ERROR("Rows[{}] has invalid schema.", i);
    }
  }
  return absl::OkStatus();
}

absl::Status Table::DoAddRows(std::vector<PartialRows>&& rows) {
  if (rows.size() != GetTableSchema()->row_schemas_.size()) {
    RUDF_RETURN_FMT_ERROR("Expected {} column set, but {} given", GetTableSchema()->row_schemas_.size(), rows.size());
  }
  // std::lock_guard<std::mutex> guard(table_mutex_);
  size_t row_count = 0;
  for (size_t i = 0; i < rows.size(); i++) {
    auto& [schema, columns] = rows[i];
    if (rows_[i].GetSchema() != *schema) {
      RUDF_RETURN_FMT_ERROR("Rows[{}] has invalid schema.", i);
    }
    if (row_count == 0) {
      row_count = columns.size();
    } else {
      if (row_count != columns.size()) {
        RUDF_RETURN_FMT_ERROR("Rows[{}] has mismatch rows size {}/{}", i, row_count, columns.size());
      }
    }
    rows_[i].Append(columns);
  }
  UnloadAllColumns();
  return absl::OkStatus();
}

absl::Status Table::InsertRow(size_t pos, const std::vector<PartialRow>& rows) {
  if (rows.size() != GetTableSchema()->row_schemas_.size()) {
    RUDF_RETURN_FMT_ERROR("Expected {} partial rows, but {} given", GetTableSchema()->row_schemas_.size(), rows.size());
  }
  // std::lock_guard<std::mutex> guard(table_mutex_);
  for (size_t i = 0; i < rows.size(); i++) {
    auto& [schema, columns] = rows[i];
    if (rows_[i].GetSchema() != *schema) {
      RUDF_RETURN_FMT_ERROR("Rows[{}] has invalid schema to insert.", i);
    }
    auto status = rows_[i].Insert(pos, columns);
    if (!status.ok()) {
      return status;
    }
  }
  UnloadAllColumns();
  return absl::OkStatus();
}

const Rows* Table::GetRowsBySchema(const RowSchema* schema) const {
  const Rows* rows = nullptr;
  for (auto& rs : rows_) {
    if (rs.GetSchema() == *schema) {
      rows = &rs;
      break;
    }
  }
  return rows;
}

VectorBase Table::LoadColumnBaseByOffset(uint32_t offset) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(this) + offset;
  VectorBase vdata = *(reinterpret_cast<const VectorBase*>(p));

  if (vdata.Size() == 0) {
    // lazy load
    uint32_t idx = GetIdxByOffset(offset);
    auto& column = columns_[idx];

    // if (column == nullptr) {
    //   THROW_LOGIC_ERR("No column found for offset:{}", offset);
    // }
    const Rows* rows = GetRowsBySchema(column->GetSchema());
    // if (column == nullptr) {
    //   THROW_LOGIC_ERR("No rows found for column:{}", column->name);
    // }

    auto status = LoadColumn(*rows, *column);
    if (!status.ok()) {
      THROW_LOGIC_ERR("Load column:{} error:{}", column->Name(), status.ToString());
    }
    vdata = *(reinterpret_cast<const VectorBase*>(p));
  }
  return vdata;
}

size_t Table::Size() const { return schema_->FieldCount(); }

size_t Table::Count() const {
  if (rows_.size() > 0) {
    return rows_[0].RowCount();
  }
  auto* first_column = reinterpret_cast<const VectorBase*>(this + 1);
  return static_cast<size_t>(first_column->Size());
}

bool Table::IsColumnLoaded(uint32_t offset) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  return vdata->Size() > 0;
}

void Table::DoFilter(Vector<Bit> bits) {
  for (auto& rows : rows_) {
    rows.Filter(bits);
  }
  UnloadAllColumns();
}

Table* Table::Filter(Vector<Bit> bits) {
  Table* new_table = Clone();
  new_table->DoFilter(bits);
  return new_table;
}
std::pair<Table*, Table*> Table::Split(Vector<Bit> bits) {
  Vector<Bit> other = ctx_.NewEmptyVector<Bit>(bits.Size());
  functions::simd_vector_bits_not(bits, other);
  Table* first = Filter(bits);
  Table* second = Filter(other);
  return {first, second};
}

Table* Table::Head(uint32_t k) {
  if (k >= Count()) {
    return this;
  }
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Truncate(k);
  }
  new_table->UnloadAllColumns();
  return new_table;
}
Table* Table::Tail(uint32_t k) {
  if (k >= Count()) {
    return this;
  }
  Table* new_table = Clone();
  new_table->UnloadAllColumns();
  for (auto& rows : new_table->rows_) {
    rows.Truncate(Count() - k, k);
  }
  return new_table;
}

template <typename T>
Table* Table::OrderBy(Vector<T> by, bool descending) {
  auto tmp_indices = GetIndices();
  Vector<int32_t> indices(tmp_indices, false);
  functions::simd_vector_sort_key_value(ctx_, by, indices, descending);
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }
  new_table->UnloadAllColumns();
  return new_table;
}
template <typename T>
Table* Table::Topk(Vector<T> by, uint32_t k, bool descending) {
  auto tmp_indices = GetIndices();
  Vector<int32_t> indices(tmp_indices, false);
  if (k > indices.Size()) {
    k = indices.Size();
  }

  functions::simd_vector_topk_key_value(ctx_, by, indices, k, descending);
  indices = indices.Resize(k);
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }
  new_table->UnloadAllColumns();
  return new_table;
}

Table* Table::GatherSubTable(std::vector<int32_t>& indices) {
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }
  new_table->ClearAllColumns();
  // Table* this_table = this;
  // schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
  //   uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
  //   if (!IsColumnLoaded(offset)) {
  //     // lazy load column
  //     return;
  //   }
  //   VectorBase new_vec = *(reinterpret_cast<VectorBase*>(vec_ptr));
  //   auto gather_result = GatherField(vec_ptr, dtype, indices);
  //   if (!gather_result.ok()) {
  //     return;
  //   }
  //   new_vec = gather_result.value();
  //   new_table->SetColumn(offset, new_vec);
  // });
  return new_table;
}

Table* Table::OrderBy(StringView column, bool descending) {
  auto result = schema_->GetField(column);
  if (!result.ok()) {
    THROW_LOGIC_ERR("No column:{} found.", column);
  }
  auto [dtype, offset] = result.value();
  uint8_t* p = reinterpret_cast<uint8_t*>(this) + offset;
  VectorBase vec_data = *(reinterpret_cast<VectorBase*>(p));
  switch (dtype.GetFundamentalType()) {
    case DATA_F64: {
      return OrderBy(Vector<double>(vec_data), descending);
    }
    case DATA_F32: {
      return OrderBy(Vector<float>(vec_data), descending);
    }
    case DATA_U64: {
      return OrderBy(Vector<uint64_t>(vec_data), descending);
    }
    case DATA_I64: {
      return OrderBy(Vector<int64_t>(vec_data), descending);
    }
    case DATA_U32: {
      return OrderBy(Vector<uint32_t>(vec_data), descending);
    }
    case DATA_I32: {
      return OrderBy(Vector<int32_t>(vec_data), descending);
    }
    default: {
      THROW_LOGIC_ERR("Invalid column:{} with dtype:{} to order_by.", column, dtype);
    }
  }
}

template <typename T>
absl::Span<Table*> Table::GroupBy(const T* by, size_t n) {
  absl::flat_hash_map<T, std::vector<int32_t>> group_idxs;
  for (size_t i = 0; i < n; i++) {
    group_idxs[by[i]].emplace_back(static_cast<int32_t>(i));
  }
  Table** group_tables = reinterpret_cast<Table**>(ctx_.ArenaAllocate(sizeof(Table*) * group_idxs.size()));
  size_t table_idx = 0;
  for (auto& [_, indices] : group_idxs) {
    Table* new_table = GatherSubTable(indices);
    group_tables[table_idx] = new_table;
    table_idx++;
  }
  return absl::Span<Table*>(group_tables, group_idxs.size());
}

template <typename T>
absl::Span<Table*> Table::GroupBy(Vector<T> by) {
  if (by.Size() != Count()) {
    THROW_LOGIC_ERR("Invalid group_by column with size:{}, while table row size:{}", by.Size(), Count());
  }
  return GroupBy(by.Data(), by.Size());
}

absl::Span<Table*> Table::GroupBy(absl::Span<const StringView> columns) {
  auto indice_table = DistinctByColumns(columns);
  Table** group_tables = reinterpret_cast<Table**>(ctx_.ArenaAllocate(sizeof(Table*) * indice_table.size()));
  size_t table_idx = 0;
  for (auto& [indice, duplicate_indices] : indice_table) {
    duplicate_indices.emplace_back(indice);
    Table* new_table = GatherSubTable(duplicate_indices);
    group_tables[table_idx] = new_table;
    table_idx++;
  }
  return absl::Span<Table*>(group_tables, indice_table.size());
}

template <typename T>
Vector<Bit> Table::Dedup(const T* data, size_t n, size_t k) {
  using DedupMap = absl::flat_hash_map<T, uint32_t>;
  DedupMap dedup_map;
  dedup_map.reserve(n);
  // size_t bits_n = n / 64;
  // if (n % 64 > 0) {
  //   bits_n++;
  // }
  Vector<Bit> result = ctx_.NewEmptyVector<Bit>(n);
  uint64_t* bits = reinterpret_cast<uint64_t*>(result.MutableData());
  // uint64_t* bits = reinterpret_cast<uint64_t*>(ctx_.ArenaAllocate(sizeof(uint64_t) * bits_n));
  for (size_t i = 0; i < n; i++) {
    size_t exist_n = dedup_map[data[i]]++;
    bits_set(bits, i, exist_n < k);
  }

  return result;
}

Vector<Bit> Table::Dedup(StringView column, uint32_t k) {
  auto result = schema_->GetField(column);
  if (!result.ok()) {
    THROW_LOGIC_ERR("No column:{} found.", column);
  }
  auto [dtype, offset] = result.value();
  size_t row_size = Count();
  VectorBase vec_data = LoadColumnBaseByOffset(offset);
  Vector<Bit> bits;
  switch (dtype.GetFundamentalType()) {
    case DATA_F64: {
      bits = Dedup(vec_data.Data<double>(), row_size, k);
      break;
    }
    case DATA_F32: {
      bits = Dedup(vec_data.Data<float>(), row_size, k);
      break;
    }
    case DATA_U64: {
      bits = Dedup(vec_data.Data<uint64_t>(), row_size, k);
      break;
    }
    case DATA_I64: {
      bits = Dedup(vec_data.Data<int64_t>(), row_size, k);
      break;
    }
    case DATA_U32: {
      bits = Dedup(vec_data.Data<uint32_t>(), row_size, k);
      break;
    }
    case DATA_I32: {
      bits = Dedup(vec_data.Data<int32_t>(), row_size, k);
      break;
    }
    case DATA_U16: {
      bits = Dedup(vec_data.Data<uint16_t>(), row_size, k);
      break;
    }
    case DATA_I16: {
      bits = Dedup(vec_data.Data<int16_t>(), row_size, k);
      break;
    }
    case DATA_U8: {
      bits = Dedup(vec_data.Data<uint8_t>(), row_size, k);
      break;
    }
    case DATA_I8: {
      bits = Dedup(vec_data.Data<int8_t>(), row_size, k);
      break;
    }
    case DATA_STRING_VIEW: {
      bits = Dedup(vec_data.Data<StringView>(), row_size, k);
      break;
    }
    default: {
      THROW_LOGIC_ERR("Invalid column:{} with dtype:{} to dedup.", column, dtype);
    }
  }

  return bits;
}

absl::Status Table::Distinct(absl::Span<const StringView> columns) {
  Vector<Bit> select;
  for (auto column : columns) {
    Vector<Bit> mask = Dedup(column, 1);
    if (select.Size() == 0) {
      select = mask;
    } else {
      functions::simd_vector_bits_and(select, mask, select);
    }
  }
  DoFilter(select);
  return absl::OkStatus();
}

typename Table::DistinctIndiceTable Table::DistinctByColumns(absl::Span<const StringView> columns) {
  size_t count = Count();
  std::vector<DType> key_dtypes;
  std::vector<VectorBase> key_vecs;
  for (auto column : columns) {
    auto result = schema_->GetField(column);
    if (!result.ok()) {
      THROW_LOGIC_ERR("No column:{} found.", column);
    }
    auto [dtype, offset] = result.value();
    VectorBase vec_data = LoadColumnBaseByOffset(offset);
    key_dtypes.emplace_back(dtype.Elem());
    key_vecs.emplace_back(vec_data);
  }
  DistinctIndiceTable indice_table;
  using DistinctTable = absl::flat_hash_map<DistinctKey, int32_t, DistinctKeyHash, DistinctKeyCompare>;
  DistinctTable distinc_table;
  distinc_table.reserve(count);

  for (size_t i = 0; i < count; i++) {
    DistinctKey key;
    for (size_t j = 0; j < key_dtypes.size(); j++) {
      key.emplace_back(std::make_pair(&key_dtypes[j], key_vecs[j].Data<uint8_t>() + key_dtypes[j].ByteSize() * i));
    }

    auto [iter, success] = distinc_table.emplace(std::move(key), static_cast<int32_t>(i));
    if (success) {
      indice_table.emplace(static_cast<int32_t>(i), std::vector<int32_t>{});
    } else {
      indice_table[iter->second].emplace_back(static_cast<int32_t>(i));
    }
  }

  return indice_table;
}

absl::Status Table::UnloadColumn(const std::string& name) {
  auto offset_result = GetColumnOffset(name);
  if (!offset_result.ok()) {
    return offset_result.status();
  }
  uint32_t offset = offset_result.value();
  uint32_t idx = GetIdxByOffset(offset);
  columns_[idx]->Unload();
  // auto* column = GetTableSchema()->GetColumnByIdx(idx);
  // if (column == nullptr || column->schema == nullptr) {
  //   RUDF_LOG_RETURN_FMT_ERROR("Invalid column:{} to unload", name);
  // }
  // uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  // VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  // vdata->SetSize(0);  // clear size for reuse
  return absl::OkStatus();
}
void Table::UnloadAllColumns() {
  for (auto& column : columns_) {
    column->Unload();
  }
}

void Table::ClearAllColumns() {
  for (auto& column : columns_) {
    column->Clear();
  }
}

const RowSchema* Table::GetRowSchemaByTypeID(uint32_t id) const { return GetTableSchema()->GetRowSchemaByTypeID(id); }

int Table::GetRowIdx(const RowSchema& schema) {
  for (size_t i = 0; i < rows_.size(); i++) {
    if (rows_[i].GetSchema() == schema) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

Table* Table::Concat(Table* other) {
  Table* new_table = Clone();
  new_table->UnloadAllColumns();
  auto status = new_table->DoConcat(other);
  if (!status.ok()) {
    THROW_LOGIC_ERR("{}", status.ToString());
  }
  return new_table;
}

absl::Status Table::DoConcat(Table* other) {
  if (GetTableSchema() != other->GetTableSchema()) {
    RUDF_RETURN_FMT_ERROR("Can NOT concat table:{} to {}", other->GetTableSchema()->Name(), GetTableSchema()->Name());
  }
  for (size_t i = 0; i < rows_.size(); i++) {
    for (size_t j = 0; j < other->rows_.size(); j++) {
      if (rows_[i].GetSchema() == other->rows_[j].GetSchema()) {
        rows_[i].Append(other->rows_[j].GetRawRowPtrs());
        break;
      }
    }
  }
  return absl::OkStatus();
}

template Table* Table::OrderBy<uint32_t>(Vector<uint32_t> by, bool descending);
template Table* Table::OrderBy<int32_t>(Vector<int32_t> by, bool descending);
template Table* Table::OrderBy<uint64_t>(Vector<uint64_t> by, bool descending);
template Table* Table::OrderBy<int64_t>(Vector<int64_t> by, bool descending);
template Table* Table::OrderBy<float>(Vector<float> by, bool descending);
template Table* Table::OrderBy<double>(Vector<double> by, bool descending);

template Table* Table::Topk<uint32_t>(Vector<uint32_t> by, uint32_t k, bool descending);
template Table* Table::Topk<int32_t>(Vector<int32_t> by, uint32_t k, bool descending);
template Table* Table::Topk<uint64_t>(Vector<uint64_t> by, uint32_t k, bool descending);
template Table* Table::Topk<int64_t>(Vector<int64_t> by, uint32_t k, bool descending);
template Table* Table::Topk<float>(Vector<float> by, uint32_t k, bool descending);
template Table* Table::Topk<double>(Vector<double> by, uint32_t k, bool descending);

template absl::Span<Table*> Table::GroupBy<double>(Vector<double> by);
template absl::Span<Table*> Table::GroupBy<float>(Vector<float> by);
template absl::Span<Table*> Table::GroupBy<uint64_t>(Vector<uint64_t> by);
template absl::Span<Table*> Table::GroupBy<int64_t>(Vector<int64_t> by);
template absl::Span<Table*> Table::GroupBy<uint32_t>(Vector<uint32_t> by);
template absl::Span<Table*> Table::GroupBy<int32_t>(Vector<int32_t> by);
template absl::Span<Table*> Table::GroupBy<uint16_t>(Vector<uint16_t> by);
template absl::Span<Table*> Table::GroupBy<int16_t>(Vector<int16_t> by);
template absl::Span<Table*> Table::GroupBy<uint8_t>(Vector<uint8_t> by);
template absl::Span<Table*> Table::GroupBy<int8_t>(Vector<int8_t> by);
template absl::Span<Table*> Table::GroupBy<StringView>(Vector<StringView> by);

}  // namespace table
}  // namespace rapidudf