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

#include "rapidudf/vector/table.h"
#include <cstring>
#include <memory>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

#include "flatbuffers/minireflect.h"

#include "rapidudf/common/allign.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/types/dyn_object_impl.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/row.h"
#include "rapidudf/vector/table_schema.h"
#include "rapidudf/vector/vector.h"

namespace rapidudf {
namespace simd {

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

void Table::Deleter::operator()(Table* ptr) const {
  ptr->~Table();
  uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
  delete[] bytes;
}

uint32_t Table::GetIdxByOffset(uint32_t offset) {
  size_t header_size = align_to<size_t>(sizeof(Table), 16);
  uint32_t idx = (offset - header_size) / sizeof(VectorData);
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
  VectorData* vdata = (reinterpret_cast<VectorData*>(vec_ptr));
  size_t request_memory_size = GetColumnMemorySize(dtype);
  if (vdata->BytesCapacity() >= request_memory_size) {
    // reuse
    return vdata->MutableData<uint8_t>();
  } else {
    uint8_t* p = ctx_.ArenaAllocate(request_memory_size);
    VectorData alloc(p, 0, request_memory_size);
    *vdata = alloc;
    return p;
  }
}
void Table::SetColumnSize(const reflect::Column& column, void* p) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + column.field.bytes_offset;
  VectorData* vdata = (reinterpret_cast<VectorData*>(vec_ptr));
  if (vdata->Data() == p) {
    vdata->SetSize(Count());
  }
}

template <typename T>
absl::Status Table::LoadColumn(const Vector<Pointer>& objs, const reflect::Column& column) {
  if (column.schema->pb_desc != nullptr) {
    return LoadProtobufColumn<T>(objs, column);
  } else if (column.schema->fbs_table != nullptr) {
    return LoadFlatbuffersColumn<T>(objs, column);
  } else {
    return LoadStructColumn<T>(objs, column);
  }
}

absl::Status Table::LoadColumn(const Rows& rows, const reflect::Column& column) {
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
    default: {
      RUDF_LOG_RETURN_FMT_ERROR("Unsupported column:{} with dtype:{}", column.name, column.field.dtype);
    }
  }
}

template <typename T>
absl::Status Table::LoadProtobufColumn(const Vector<Pointer>& pb_vector, const reflect::Column& column) {
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
absl::Status Table::LoadFlatbuffersColumn(const Vector<Pointer>& fbs_vector, const reflect::Column& column) {
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
absl::Status Table::LoadStructColumn(const Vector<Pointer>& struct_vector, const reflect::Column& column) {
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
            vec[i] = (*ptr);
          } else {
            vec[i] = (T{});
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
            vec[i] = (f(obj.As<uint8_t>()));
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
std::vector<int32_t> Table::GetIndices() {
  size_t count = Count();
  if (indices_.size() < count) {
    // indices_.resize(count);
    // std::iota(indices_.begin(), indices_.end(), 0);
    indices_ = get_indices(count);
  } else if (indices_.size() > count) {
    indices_.resize(count);
  }
  return indices_;
}
void Table::SetIndices(std::vector<int32_t>&& indices) { indices_ = std::move(indices); }

absl::Status Table::DoAddRows(std::vector<PartialRows>&& rows) {
  if (rows.size() != GetTableSchema()->row_schemas_.size()) {
    RUDF_RETURN_FMT_ERROR("Expected {} column set, but {} given", GetTableSchema()->row_schemas_.size(), rows.size());
  }
  size_t row_count = 0;
  for (auto& [schema, columns] : rows) {
    if (row_count == 0) {
      row_count = columns.size();
    } else {
      if (row_count != columns.size()) {
        RUDF_RETURN_FMT_ERROR("Mismatch rows size {}/{}", row_count, columns.size());
      }
    }
    bool appened = false;
    for (auto& rows : rows_) {
      if (rows.GetSchema() == *schema) {
        rows.Append(columns);
        appened = true;
        break;
      }
    }
    if (!appened) {
      RUDF_RETURN_FMT_ERROR("Missing schema to add rows");
    }
  }
  return absl::OkStatus();
}

absl::Status Table::DoAddRows(std::vector<const uint8_t*>&& row_objs, const RowSchema& schema) {
  if (!GetTableSchema()->ExistRow(schema)) {
    RUDF_RETURN_FMT_ERROR("Unsupported schema to add rows");
  }
  for (auto& rows : rows_) {
    if (rows.GetSchema() == schema) {
      rows.Append(row_objs);
      return absl::OkStatus();
    }
  }
  rows_.emplace_back(Rows(ctx_, std::move(row_objs), schema));
  return absl::OkStatus();
}
absl::Status Table::InsertRow(size_t pos, const std::vector<PartialRow>& row) {
  if (row.size() != GetTableSchema()->row_schemas_.size()) {
    RUDF_RETURN_FMT_ERROR("Expected {} partial rows, but {} given", GetTableSchema()->row_schemas_.size(), row.size());
  }
  for (auto& [schema, partial_row] : row) {
    bool inserted = false;
    for (auto& rows : rows_) {
      if (rows.GetSchema() == *schema) {
        auto status = rows.Insert(pos, partial_row);
        if (!status.ok()) {
          return status;
        }
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      RUDF_RETURN_FMT_ERROR("Missing schema to insert row");
    }
  }
  return absl::OkStatus();
}

VectorData Table::GetColumnByOffset(uint32_t offset) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(this) + offset;
  VectorData vdata = *(reinterpret_cast<const VectorData*>(p));

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
    vdata = *(reinterpret_cast<const VectorData*>(p));
  }

  return vdata;
}

void Table::SetColumn(uint32_t offset, VectorData vec) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  *(reinterpret_cast<VectorData*>(vec_ptr)) = vec;
}

absl::StatusOr<VectorData> Table::GatherField(uint8_t* vec_ptr, const DType& dtype, Vector<int32_t> indices) {
  VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
  switch (dtype.GetFundamentalType()) {
    case DATA_BIT: {
      new_vec =
          functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), indices).GetVectorData();
      break;
    }
    case DATA_U8: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint8_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_U16: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint16_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_U32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint32_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_U64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint64_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_I8: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int8_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_I16: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int16_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_I32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int32_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_I64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int64_t>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_F32: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<float>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_F64: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<double>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_STRING_VIEW: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<StringView>*>(vec_ptr), indices)
                    .GetVectorData();
      break;
    }
    case DATA_POINTER: {
      new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Pointer>*>(vec_ptr), indices)
                    .GetVectorData();
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
  auto* first_column = reinterpret_cast<const VectorData*>(this + 1);
  return first_column->Size();
}

bool Table::IsColumnLoaded(uint32_t offset) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  VectorData* vdata = (reinterpret_cast<VectorData*>(vec_ptr));
  return vdata->Size() > 0;
}

Table* Table::Filter(Vector<Bit> bits) {
  Table* this_table = this;
  Table* new_table = Clone();

  for (auto& rows : new_table->rows_) {
    rows.Filter(bits);
  }
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    if (!IsColumnLoaded(offset)) {
      // lazy load column
      return;
    }

    VectorData new_vec;
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec =
            functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), bits).GetVectorData();
        break;
      }
      case DATA_U8: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint8_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U16: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint16_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U32: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint32_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint64_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I8: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int8_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I16: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int16_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I32: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int32_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int64_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_F32: {
        new_vec =
            functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<float>*>(vec_ptr), bits).GetVectorData();
        break;
      }
      case DATA_F64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<double>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_STRING_VIEW: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<StringView>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      default: {
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}
Table* Table::Head(uint32_t k) {
  if (k >= Count()) {
    return this;
  }
  Table* this_table = this;
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Truncate(k);
  }
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    if (!IsColumnLoaded(offset)) {
      // lazy load column
      return;
    }
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (k < new_vec.Size()) {
      new_vec.SetSize(k);
    }
    new_table->SetColumn(offset, new_vec);
  });
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
  Vector<int32_t> indices(tmp_indices);
  functions::simd_vector_sort_key_value(ctx_, by, indices, descending);
  Table* this_table = this;
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }

  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    if (!IsColumnLoaded(offset)) {
      // lazy load column
      return;
    }
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_table->SetColumn(offset, new_vec);
      return;
    }
    auto gather_result = GatherField(vec_ptr, dtype, indices);
    if (!gather_result.ok()) {
      return;
    }
    new_vec = gather_result.value();
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}
template <typename T>
Table* Table::Topk(Vector<T> by, uint32_t k, bool descending) {
  auto tmp_indices = GetIndices();
  Vector<int32_t> indices(tmp_indices);
  if (k > indices.Size()) {
    k = indices.Size();
  }

  functions::simd_vector_topk_key_value(ctx_, by, indices, k, descending);
  indices = indices.Resize(k);
  Table* this_table = this;
  Table* new_table = Clone();
  for (auto& rows : new_table->rows_) {
    rows.Gather(indices);
  }
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData vdata = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (vdata.Data() == nullptr) {
      // lazy load column
      return;
    }
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_vec.SetSize(k);
      new_table->SetColumn(offset, new_vec);
      return;
    }
    auto gather_result = GatherField(vec_ptr, dtype, indices);
    if (!gather_result.ok()) {
      return;
    }
    new_vec = gather_result.value();
    new_table->SetColumn(offset, new_vec);
  });

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
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
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
  VectorData vec_data = *(reinterpret_cast<VectorData*>(p));
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

template <typename T>
absl::Span<Table*> Table::GroupBy(Vector<T> by) {
  if (by.Size() != Count()) {
    THROW_LOGIC_ERR("Invalid group_by column with size:{}, while table row size:{}", by.Size(), Count());
  }
  return GroupBy(by.Data(), by.Size());
}

absl::Span<Table*> Table::GroupBy(StringView column) {
  auto result = schema_->GetField(column);
  if (!result.ok()) {
    THROW_LOGIC_ERR("No column:{} found.", column);
  }
  auto [dtype, offset] = result.value();
  size_t row_size = Count();
  uint8_t* p = reinterpret_cast<uint8_t*>(this) + offset;
  VectorData vec_data = *(reinterpret_cast<VectorData*>(p));
  switch (dtype.GetFundamentalType()) {
    case DATA_F64: {
      return GroupBy(reinterpret_cast<const double*>(vec_data.Data()), row_size);
    }
    case DATA_F32: {
      return GroupBy(reinterpret_cast<const float*>(vec_data.Data()), row_size);
    }
    case DATA_U64: {
      return GroupBy(reinterpret_cast<const uint64_t*>(vec_data.Data()), row_size);
    }
    case DATA_I64: {
      return GroupBy(reinterpret_cast<const int64_t*>(vec_data.Data()), row_size);
    }
    case DATA_U32: {
      return GroupBy(reinterpret_cast<const uint32_t*>(vec_data.Data()), row_size);
    }
    case DATA_I32: {
      return GroupBy(reinterpret_cast<const int32_t*>(vec_data.Data()), row_size);
    }
    case DATA_U16: {
      return GroupBy(reinterpret_cast<const uint16_t*>(vec_data.Data()), row_size);
    }
    case DATA_I16: {
      return GroupBy(reinterpret_cast<const int16_t*>(vec_data.Data()), row_size);
    }
    case DATA_U8: {
      return GroupBy(reinterpret_cast<const uint8_t*>(vec_data.Data()), row_size);
    }
    case DATA_I8: {
      return GroupBy(reinterpret_cast<const int8_t*>(vec_data.Data()), row_size);
    }
    case DATA_STRING_VIEW: {
      return GroupBy(reinterpret_cast<const StringView*>(vec_data.Data()), row_size);
    }
    default: {
      THROW_LOGIC_ERR("Invalid column:{} with dtype:{} to group_by.", column, dtype);
    }
  }
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
  VectorData* vdata = (reinterpret_cast<VectorData*>(vec_ptr));
  vdata->SetSize(0);  // clear size for reuse
  return absl::OkStatus();
}
void Table::UnloadAllColumns() {
  for (auto& column : GetTableSchema()->columns_) {
    if (column.schema != nullptr) {
      uint32_t offset = column.field.bytes_offset;
      uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
      VectorData* vdata = (reinterpret_cast<VectorData*>(vec_ptr));
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
}  // namespace simd
}  // namespace rapidudf