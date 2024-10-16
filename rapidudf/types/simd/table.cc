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

#include "rapidudf/types/simd/table.h"
#include <cstring>
#include <type_traits>
#include <vector>
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/types/dyn_object_impl.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace simd {

TableSchema* TableSchema::GetMutable(const std::string& name) {
  DynObjectSchema* s = DynObjectSchema::GetMutable(name);
  return reinterpret_cast<TableSchema*>(s);
}

const TableSchema* TableSchema::Get(const std::string& name) { return GetMutable(name); }

bool TableSchema::ExistColumn(const std::string& name) const { return DynObjectSchema::ExistField(name); }

const TableSchema* TableSchema::GetOrCreate(const std::string& name, InitFunc&& init) {
  const DynObjectSchema* s = DynObjectSchema::GetOrCreate(
      name,
      [&](DynObjectSchema* s) {
        TableSchema* table_schema = reinterpret_cast<TableSchema*>(s);
        init(table_schema);
        Flags flags(true);
        table_schema->SetFlags(flags);
      },
      sizeof(Table));
  return reinterpret_cast<const TableSchema*>(s);
}

typename Table::SmartPtr TableSchema::NewTable(Context& ctx) const {
  uint8_t* bytes = new uint8_t[ByteSize()];
  memset(bytes, 0, ByteSize());
  try {
    new (bytes) Table(ctx, this);
  } catch (...) {
    throw;
  }
  Table::SmartPtr p(reinterpret_cast<Table*>(bytes));
  return p;
}

absl::Status TableSchema::BuildFromProtobuf(const ::google::protobuf::Message* msg) {
  const ::google::protobuf::Descriptor* desc = msg->GetDescriptor();
  for (int i = 0; i < desc->field_count(); i++) {
    const ::google::protobuf::FieldDescriptor* field_desc = desc->field(i);
    absl::Status status;
    switch (field_desc->type()) {
      case ::google::protobuf::FieldDescriptor::TYPE_BOOL: {
        status = AddColumn<bool>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_DOUBLE: {
        status = AddColumn<double>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FLOAT: {
        status = AddColumn<float>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_INT64: {
        status = AddColumn<int64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_UINT64: {
        status = AddColumn<uint64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_INT32: {
        status = AddColumn<int32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_UINT32: {
        status = AddColumn<uint32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED64: {
        status = AddColumn<uint64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED32: {
        status = AddColumn<uint32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED64: {
        status = AddColumn<int64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED32: {
        status = AddColumn<int32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SINT64: {
        status = AddColumn<int64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SINT32: {
        status = AddColumn<int32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_STRING:
      case ::google::protobuf::FieldDescriptor::TYPE_BYTES: {
        status = AddColumn<std::string>(field_desc->name());
        break;
      }
      default: {
        status = absl::UnimplementedError(fmt::format("Not supported field:{} with message type", field_desc->name()));
        break;
      }
    }
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status Table::SetColumnByProtobufField(const std::vector<const ::google::protobuf::Message*>& pb_vector,
                                             const ::google::protobuf::Reflection* reflect,
                                             const ::google::protobuf::FieldDescriptor* field) {
  std::vector<T> vec;
  vec.reserve(pb_vector.size());

  for (auto* msg : pb_vector) {
    if constexpr (std::is_same_v<bool, T>) {
      vec.emplace_back(reflect->GetBool(*msg, field));
    } else if constexpr (std::is_same_v<int32_t, T>) {
      vec.emplace_back(reflect->GetInt32(*msg, field));
    } else if constexpr (std::is_same_v<int64_t, T>) {
      vec.emplace_back(reflect->GetInt64(*msg, field));
    } else if constexpr (std::is_same_v<uint32_t, T>) {
      vec.emplace_back(reflect->GetUInt32(*msg, field));
    } else if constexpr (std::is_same_v<uint64_t, T>) {
      vec.emplace_back(reflect->GetUInt64(*msg, field));
    } else if constexpr (std::is_same_v<float, T>) {
      vec.emplace_back(reflect->GetFloat(*msg, field));
    } else if constexpr (std::is_same_v<double, T>) {
      vec.emplace_back(reflect->GetDouble(*msg, field));
    } else if constexpr (std::is_same_v<std::string, T>) {
      vec.emplace_back(reflect->GetString(*msg, field));
    } else {
    }
  }
  return Set(field->name(), std::move(vec));
  // return absl::OkStatus();
}

absl::Status Table::BuildFromProtobufVector(const std::vector<const ::google::protobuf::Message*>& pb_vector) {
  const ::google::protobuf::Descriptor* desc = pb_vector[0]->GetDescriptor();
  const ::google::protobuf::Reflection* reflect = pb_vector[0]->GetReflection();

  for (int i = 0; i < desc->field_count(); i++) {
    const ::google::protobuf::FieldDescriptor* field_desc = desc->field(i);
    absl::Status status;
    switch (field_desc->type()) {
      case ::google::protobuf::FieldDescriptor::TYPE_BOOL: {
        status = SetColumnByProtobufField<bool>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_DOUBLE: {
        status = SetColumnByProtobufField<double>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FLOAT: {
        status = SetColumnByProtobufField<float>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_INT64:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED64:
      case ::google::protobuf::FieldDescriptor::TYPE_SINT64: {
        status = SetColumnByProtobufField<int64_t>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_UINT64:
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED64: {
        status = SetColumnByProtobufField<uint64_t>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_INT32:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED32:
      case ::google::protobuf::FieldDescriptor::TYPE_SINT32: {
        status = SetColumnByProtobufField<int32_t>(pb_vector, reflect, field_desc);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_UINT32:
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED32: {
        status = SetColumnByProtobufField<uint32_t>(pb_vector, reflect, field_desc);
        break;
      }

      case ::google::protobuf::FieldDescriptor::TYPE_STRING:
      case ::google::protobuf::FieldDescriptor::TYPE_BYTES: {
        status = SetColumnByProtobufField<std::string>(pb_vector, reflect, field_desc);
        break;
      }
      default: {
        status = absl::UnimplementedError(fmt::format("Not supported field:{} with message type", field_desc->name()));
        break;
      }
    }
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

Table* Table::Clone() {
  uint8_t* new_table = reinterpret_cast<uint8_t*>(ctx_.ArenaAllocate(schema_->ByteSize()));
  uint32_t bytes = schema_->ByteSize();
  memcpy(new_table, this, bytes);
  Table* t = reinterpret_cast<Table*>(new_table);
  absl::Status status;
  return t;
}
Vector<int32_t> Table::GetIndices() {
  if (indices_.Size() == 0) {
    auto* first_column = reinterpret_cast<VectorData*>(this + 1);
    indices_ = functions::simd_vector_iota<int32_t>(ctx_, 0, first_column->Size());
  }
  auto* p = ctx_.ArenaAllocate(sizeof(int32_t) * indices_.Size());
  memcpy(p, indices_.Data(), sizeof(int32_t) * indices_.Size());
  VectorData vdata(p, indices_.Size(), sizeof(int32_t) * indices_.Size());
  vdata.SetReadonly(false);
  return Vector<int32_t>(vdata);
}

void Table::SetColumn(uint32_t offset, VectorData vec) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  memcpy(vec_ptr, &vec, sizeof(vec));
}
void Table::SetSize(uint32_t k) {}

size_t Table::Size() const { return schema_->FieldCount(); }

Table* Table::Filter(Vector<Bit> bits) {
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
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
Table* Table::Take(uint32_t k) {
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (k < new_vec.Size()) {
      new_vec.SetSize(k);
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}

template <typename T>
Table* Table::OrderBy(Vector<T> by, bool descending) {
  auto indices = GetIndices();
  functions::simd_vector_sort_key_value(ctx_, by, indices, descending);
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_table->SetColumn(offset, new_vec);
      return;
    }
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), indices)
                      .GetVectorData();
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
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}
template <typename T>
Table* Table::Topk(Vector<T> by, uint32_t k, bool descending) {
  auto indices = GetIndices();
  if (k > indices.Size()) {
    k = indices.Size();
  }
  functions::simd_vector_topk_key_value(ctx_, by, indices, k, descending);
  indices = indices.Resize(k);
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_vec.SetSize(k);
      new_table->SetColumn(offset, new_vec);
      return;
    }
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), indices)
                      .GetVectorData();
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
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  if (new_table->indices_.Size() > 0) {
    new_table->indices_.Resize(k);
  }
  return new_table;
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

}  // namespace simd
}  // namespace rapidudf