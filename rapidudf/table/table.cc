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
}
Table::Table(Table& other) : DynObject(other), ctx_(other.ctx_) {
  indices_ = other.indices_;
  for (auto& row : other.rows_) {
    auto ptrs = row.GetRawRowPtrs();
    rows_.emplace_back(Rows(ctx_, std::move(ptrs), row.GetSchema()));
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
size_t Table::GetColumnMemorySize(const DType& dtype) {
  if (dtype.IsBit()) {
    uint32_t n = Count() / 8;
    return Count() % 8 > 0 ? n + 1 : n;
  } else {
    return dtype.ByteSize() * Count();
  }
}
uint8_t* Table::GetColumnMemory(uint32_t offset, const DType& dtype) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  size_t request_memory_size = GetColumnMemorySize(dtype);

  if (vdata->Capacity() >= request_memory_size) {
    // reuse
    return vdata->MutableData<uint8_t>();
  } else {
    uint8_t* p = ctx_.ArenaAllocate(request_memory_size);
    VectorBase alloc(p, 0, Count(), false);
    *vdata = alloc;
    return p;
  }
}
void Table::SetColumnSize(const Column& column, void* p) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + column.field.bytes_offset;
  VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  if (vdata->Data<void>() == p) {
    vdata->SetSize(Count());
  }
  vdata->SetWritable();
}

template <typename T>
absl::Status Table::LoadColumn(const Vector<Pointer>& objs, const Column& column) {
  if (column.schema->pb_desc != nullptr) {
    return LoadProtobufColumn<T>(objs, column);
  } else if (column.schema->fbs_table != nullptr) {
    return LoadFlatbuffersColumn<T>(objs, column);
  } else {
    return LoadStructColumn<T>(objs, column);
  }
}

absl::Status Table::LoadColumn(const Rows& rows, const Column& column) {
  switch (column.field.dtype.Elem().GetFundamentalType()) {
    case DATA_F32: {
      return LoadColumn<float>(rows.GetRowPtrs(), column);
    }
    case DATA_F64: {
      return LoadColumn<double>(rows.GetRowPtrs(), column);
    }
    case DATA_U64: {
      return LoadColumn<uint64_t>(rows.GetRowPtrs(), column);
    }
    case DATA_U32: {
      return LoadColumn<uint32_t>(rows.GetRowPtrs(), column);
    }
    case DATA_U16: {
      return LoadColumn<uint16_t>(rows.GetRowPtrs(), column);
    }
    case DATA_U8: {
      return LoadColumn<uint8_t>(rows.GetRowPtrs(), column);
    }
    case DATA_I64: {
      return LoadColumn<int64_t>(rows.GetRowPtrs(), column);
    }
    case DATA_I32: {
      return LoadColumn<int32_t>(rows.GetRowPtrs(), column);
    }
    case DATA_I16: {
      return LoadColumn<int16_t>(rows.GetRowPtrs(), column);
    }
    case DATA_I8: {
      return LoadColumn<int8_t>(rows.GetRowPtrs(), column);
    }
    case DATA_STRING_VIEW: {
      return LoadColumn<StringView>(rows.GetRowPtrs(), column);
    }
    case DATA_BIT: {
      return LoadColumn<bool>(rows.GetRowPtrs(), column);
    }
    default: {
      RUDF_LOG_RETURN_FMT_ERROR("Unsupported column:{} with dtype:{}", column.name, column.field.dtype);
    }
  }
}

template <typename T>
absl::Status Table::LoadProtobufColumn(const Vector<Pointer>& pb_vector, const Column& column) {
  T* vec = reinterpret_cast<T*>(GetColumnMemory(column.field.bytes_offset, get_dtype<T>()));
  const ::google::protobuf::FieldDescriptor* field_desc = column.GetProtobufField();
  for (size_t i = 0; i < pb_vector.Size(); i++) {
    auto obj = pb_vector[i];
    if (!obj.IsNull()) {
      const ::google::protobuf::Message* msg = obj.As<const ::google::protobuf::Message>();
      const ::google::protobuf::Reflection* reflect = msg->GetReflection();
      if constexpr (std::is_same_v<bool, T>) {
        vec[i] = (reflect->GetBool(*msg, field_desc));
      } else if constexpr (std::is_same_v<int32_t, T>) {
        vec[i] = (reflect->GetInt32(*msg, field_desc));
      } else if constexpr (std::is_same_v<int64_t, T>) {
        vec[i] = (reflect->GetInt64(*msg, field_desc));
      } else if constexpr (std::is_same_v<uint32_t, T>) {
        vec[i] = (reflect->GetUInt32(*msg, field_desc));
      } else if constexpr (std::is_same_v<uint64_t, T>) {
        vec[i] = (reflect->GetUInt64(*msg, field_desc));
      } else if constexpr (std::is_same_v<float, T>) {
        vec[i] = (reflect->GetFloat(*msg, field_desc));
      } else if constexpr (std::is_same_v<double, T>) {
        vec[i] = (reflect->GetDouble(*msg, field_desc));
      } else if constexpr (std::is_same_v<StringView, T>) {
        static std::string empty;
        const std::string& ref = reflect->GetStringReference(*msg, field_desc, &empty);
        if (!ref.empty()) {
          vec[i] = (StringView(ref));
        } else {
          vec[i] = (StringView());
        }
      } else {
        RUDF_LOG_RETURN_FMT_ERROR("Unsupported pb field with type:{}/{}", msg->GetTypeName(),
                                  field_desc->cpp_type_name());
      }
    } else {
      vec[i] = (T{});
    }
  }

  SetColumnSize(column, vec);
  return absl::OkStatus();
}
template <typename T>
absl::Status Table::LoadFlatbuffersColumn(const Vector<Pointer>& fbs_vector, const Column& column) {
  T* vec = reinterpret_cast<T*>(GetColumnMemory(column.field.bytes_offset, get_dtype<T>()));
  for (size_t i = 0; i < fbs_vector.Size(); i++) {
    auto fbs = fbs_vector[i];
    const uint8_t* ptr = nullptr;
    if (!fbs.IsNull()) {
      ptr = fbs.As<const flatbuffers::Table>()->GetAddressOf(
          flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(column.field_idx)));
    }
    if (ptr == nullptr) {
      vec[i] = (T{});
      continue;
    }
    if constexpr (std::is_same_v<bool, T> || std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T> ||
                  std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint32_t, T> ||
                  std::is_same_v<int32_t, T> || std::is_same_v<int64_t, T> || std::is_same_v<uint64_t, T> ||
                  std::is_same_v<float, T> || std::is_same_v<double, T>) {
      vec[i] = (flatbuffers::ReadScalar<T>(ptr));
    } else if constexpr (std::is_same_v<StringView, T>) {
      ptr += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(ptr);
      const flatbuffers::String* str = reinterpret_cast<const flatbuffers::String*>(ptr);
      StringView s(reinterpret_cast<const char*>(str->Data()), str->size());
      vec[i] = (s);
    }
  }

  SetColumnSize(column, vec);
  return absl::OkStatus();
}

template <typename T>
absl::Status Table::LoadStructColumn(const Vector<Pointer>& struct_vector, const Column& column) {
  DType expect_dtype = get_dtype<T>();
  DType actual_dtype;
  T* vec = reinterpret_cast<T*>(GetColumnMemory(column.field.bytes_offset, expect_dtype));
  if (column.GetStructField()->HasField()) {
    actual_dtype = *(column.GetStructField()->member_field_dtype);
  } else {
    actual_dtype = (column.GetStructField()->member_func->return_type);
  }

  if (column.GetStructField()->HasField()) {
    for (size_t i = 0; i < struct_vector.Size(); i++) {
      auto obj = struct_vector[i];
      if constexpr (std::is_same_v<bool, T> || std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T> ||
                    std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint32_t, T> ||
                    std::is_same_v<int32_t, T> || std::is_same_v<int64_t, T> || std::is_same_v<uint64_t, T> ||
                    std::is_same_v<float, T> || std::is_same_v<double, T>) {
        if (expect_dtype == actual_dtype) {
          if (!obj.IsNull()) {
            const T* ptr =
                reinterpret_cast<const T*>(obj.As<const uint8_t>() + column.GetStructField()->member_field_offset);
            if constexpr (std::is_same_v<bool, T>) {
              bits_set(reinterpret_cast<uint8_t*>(vec), i, *ptr);
            } else {
              vec[i] = (*ptr);
            }
          } else {
            if constexpr (std::is_same_v<bool, T>) {
              bits_set(reinterpret_cast<uint8_t*>(vec), i, false);
            } else {
              vec[i] = (T{});
            }
          }
        } else {
          RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype,
                                    actual_dtype);
        }
      } else if constexpr (std::is_same_v<StringView, T>) {
        if (!obj.IsNull()) {
          if (expect_dtype == actual_dtype) {
            const T* ptr =
                reinterpret_cast<const T*>(obj.As<const uint8_t>() + column.GetStructField()->member_field_offset);
            vec[i] = (*ptr);
          } else if (actual_dtype.IsString()) {
            const std::string* ptr = reinterpret_cast<const std::string*>(obj.As<const uint8_t>() +
                                                                          column.GetStructField()->member_field_offset);
            vec[i] = (StringView(*ptr));
          } else if (actual_dtype.IsStdStringView()) {
            const std::string_view* ptr = reinterpret_cast<const std::string_view*>(
                obj.As<const uint8_t>() + column.GetStructField()->member_field_offset);
            vec[i] = (StringView(*ptr));
          } else {
            RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype,
                                      actual_dtype);
          }
        } else {
          vec[i] = (T{});
        }
      } else {
        RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype, actual_dtype);
      }
    }
  } else {
    for (size_t i = 0; i < struct_vector.Size(); i++) {
      auto obj = struct_vector[i];
      if (!obj.IsNull()) {
        if constexpr (std::is_same_v<bool, T> || std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T> ||
                      std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint32_t, T> ||
                      std::is_same_v<int32_t, T> || std::is_same_v<int64_t, T> || std::is_same_v<uint64_t, T> ||
                      std::is_same_v<float, T> || std::is_same_v<double, T>) {
          if (expect_dtype == actual_dtype) {
            using func_t = T (*)(void*);
            func_t f = reinterpret_cast<func_t>(column.GetStructField()->member_func->func);
            if constexpr (std::is_same_v<bool, T>) {
              bits_set(reinterpret_cast<uint8_t*>(vec), i, f(obj.As<uint8_t>()));
            } else {
              vec[i] = (f(obj.As<uint8_t>()));
            }
          } else {
            RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype,
                                      actual_dtype);
          }
        } else if constexpr (std::is_same_v<StringView, T>) {
          if (expect_dtype == actual_dtype) {
            using func_t = T (*)(void*);
            func_t f = reinterpret_cast<func_t>(column.GetStructField()->member_func->func);
            vec[i] = (f(obj.As<uint8_t>()));
          } else if (actual_dtype.IsStringPtr()) {
            using func_t = const std::string& (*)(void*);
            func_t f = reinterpret_cast<func_t>(column.GetStructField()->member_func->func);
            vec[i] = (StringView(f(obj.As<uint8_t>())));
          } else if (actual_dtype.IsStdStringView()) {
            using func_t = std::string_view (*)(void*);
            func_t f = reinterpret_cast<func_t>(column.GetStructField()->member_func->func);
            vec[i] = (StringView(f(obj.As<uint8_t>())));
          } else {
            RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype,
                                      actual_dtype);
          }
        } else {
          RUDF_LOG_RETURN_FMT_ERROR("Unexpected state with column dtype:{}, field dtype:{}", expect_dtype,
                                    actual_dtype);
        }
      } else {
        vec[i] = (T{});
      }
    }
  }

  SetColumnSize(column, vec);
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
template <typename T>
Vector<T> Table::LoadColumnByOffset(uint32_t offset) {
  VectorBase base = LoadColumnBaseByOffset(offset);
  return *(reinterpret_cast<Vector<T>*>(&base));
}

VectorBase Table::LoadColumnBaseByOffset(uint32_t offset) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(this) + offset;
  VectorBase vdata = *(reinterpret_cast<const VectorBase*>(p));

  if (vdata.Size() == 0) {
    // lazy load
    uint32_t idx = GetIdxByOffset(offset);
    auto* column = GetTableSchema()->GetColumnByIdx(idx);

    if (column == nullptr) {
      THROW_LOGIC_ERR("No column found for offset:{}", offset);
    }
    const Rows* rows = nullptr;
    for (auto& rs : rows_) {
      if (rs.GetSchema() == *(column->schema)) {
        rows = &rs;
        break;
      }
    }
    if (column == nullptr) {
      THROW_LOGIC_ERR("No rows found for column:{}", column->name);
    }

    auto status = LoadColumn(*rows, *column);
    if (!status.ok()) {
      THROW_LOGIC_ERR("Load column:{} error:{}", column->name, status.ToString());
    }
    vdata = *(reinterpret_cast<const VectorBase*>(p));
  }
  return vdata;
}

void Table::SetColumn(uint32_t offset, VectorBase vec) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  *(reinterpret_cast<VectorBase*>(vec_ptr)) = vec;
}

absl::StatusOr<VectorBase> Table::GatherField(uint8_t* vec_ptr, const DType& dtype, Vector<int32_t> indices) {
  VectorBase new_vec = *(reinterpret_cast<VectorBase*>(vec_ptr));
  switch (dtype.GetFundamentalType()) {
    case DATA_BIT: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<Bit>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_U8: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<uint8_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_U16: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<uint16_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_U32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<uint32_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_U64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<uint64_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_I8: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<int8_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_I16: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<int16_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_I32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<int32_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_I64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<int64_t>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_F32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<float>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_F64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<double>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_STRING_VIEW: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<StringView>*>(vec_ptr), indices).RawData();
      break;
    }
    case DATA_POINTER: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<Vector<Pointer>*>(vec_ptr), indices).RawData();
      break;
    }
    default: {
      RUDF_LOG_RETURN_FMT_ERROR("Unsupported fielddtype:{} togather", dtype);
    }
  }
  return new_vec;
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

Table* Table::SubTable(std::vector<int32_t>& indices) {
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }
  Table* this_table = this;
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    if (!IsColumnLoaded(offset)) {
      // lazy load column
      return;
    }
    VectorBase new_vec = *(reinterpret_cast<VectorBase*>(vec_ptr));
    auto gather_result = GatherField(vec_ptr, dtype, indices);
    if (!gather_result.ok()) {
      return;
    }
    new_vec = gather_result.value();
    new_table->SetColumn(offset, new_vec);
  });
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
    Table* new_table = SubTable(indices);
    group_tables[table_idx] = new_table;
    table_idx++;
  }
  return absl::Span<Table*>(group_tables, group_idxs.size());
}

// VectorBuf Table::GetColumnVectorBuf(StringView column) {
//   auto result = schema_->GetField(column);
//   if (!result.ok()) {
//     THROW_LOGIC_ERR("No column:{} found.", column);
//   }
//   auto [dtype, offset] = result.value();
//   uint8_t* p = reinterpret_cast<uint8_t*>(this) + offset;
//   VectorBuf vec_data = *(reinterpret_cast<VectorBuf*>(p));
//   if (vec_data.Data() == nullptr) {
//     return GetColumnByOffset(offset);
//   }
//   return vec_data;
// }

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
    Table* new_table = SubTable(duplicate_indices);
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
  // VectorBuf vdata(bits, n, sizeof(uint64_t) * bits_n);
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
  auto* column = GetTableSchema()->GetColumnByIdx(idx);
  if (column == nullptr || column->schema == nullptr) {
    RUDF_LOG_RETURN_FMT_ERROR("Invalid column:{} to unload", name);
  }
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
  vdata->SetSize(0);  // clear size for reuse
  return absl::OkStatus();
}
void Table::UnloadAllColumns() {
  for (auto& column : GetTableSchema()->columns_) {
    if (column.schema != nullptr) {
      uint32_t offset = column.field.bytes_offset;
      uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
      VectorBase* vdata = (reinterpret_cast<VectorBase*>(vec_ptr));
      vdata->SetSize(0);  // clear size for reuse
    }
  }
}

const RowSchema* Table::GetRowSchema(const RowSchema& schema) {
  for (auto& s : GetTableSchema()->row_schemas_) {
    if (*s == schema) {
      return s.get();
    }
  }
  return nullptr;
}
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

template Vector<double> Table::LoadColumnByOffset<double>(uint32_t);
template Vector<float> Table::LoadColumnByOffset<float>(uint32_t);
template Vector<uint64_t> Table::LoadColumnByOffset<uint64_t>(uint32_t);
template Vector<int64_t> Table::LoadColumnByOffset<int64_t>(uint32_t);
template Vector<uint32_t> Table::LoadColumnByOffset<uint32_t>(uint32_t);
template Vector<int32_t> Table::LoadColumnByOffset<int32_t>(uint32_t);
template Vector<uint16_t> Table::LoadColumnByOffset<uint16_t>(uint32_t);
template Vector<int16_t> Table::LoadColumnByOffset<int16_t>(uint32_t);
template Vector<uint8_t> Table::LoadColumnByOffset<uint8_t>(uint32_t);
template Vector<int8_t> Table::LoadColumnByOffset<int8_t>(uint32_t);
template Vector<StringView> Table::LoadColumnByOffset<StringView>(uint32_t);
template Vector<Bit> Table::LoadColumnByOffset<Bit>(uint32_t);
template Vector<Pointer> Table::LoadColumnByOffset<Pointer>(uint32_t);
}  // namespace table
}  // namespace rapidudf