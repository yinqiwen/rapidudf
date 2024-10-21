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

#include "rapidudf/types/simd/table.h"
#include <cstring>
#include <type_traits>
#include <vector>

#include "flatbuffers/minireflect.h"

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

bool TableCreateOptions::IsAllowed(const std::string& field) const {
  if (!includes.empty()) {
    return includes.count(field) == 1;
  }
  if (!excludes.empty()) {
    return excludes.count(field) == 0;
  }
  return true;
}

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

absl::Status TableSchema::BuildFromProtobuf(const ::google::protobuf::Message* msg, const TableCreateOptions& opts) {
  const ::google::protobuf::Descriptor* desc = msg->GetDescriptor();
  for (int i = 0; i < desc->field_count(); i++) {
    const ::google::protobuf::FieldDescriptor* field_desc = desc->field(i);
    if (!opts.IsAllowed(field_desc->name())) {
      continue;
    }
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
      case ::google::protobuf::FieldDescriptor::TYPE_SINT64:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED64:
      case ::google::protobuf::FieldDescriptor::TYPE_INT64: {
        status = AddColumn<int64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED64:
      case ::google::protobuf::FieldDescriptor::TYPE_UINT64: {
        status = AddColumn<uint64_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SINT32:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED32:
      case ::google::protobuf::FieldDescriptor::TYPE_INT32: {
        status = AddColumn<int32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED32:
      case ::google::protobuf::FieldDescriptor::TYPE_UINT32: {
        status = AddColumn<uint32_t>(field_desc->name());
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_STRING:
      case ::google::protobuf::FieldDescriptor::TYPE_BYTES: {
        status = AddColumn<std::string>(field_desc->name());
        break;
      }
      default: {
        if (opts.ignore_unsupported_fields) {
          status = absl::OkStatus();
        } else {
          status =
              absl::UnimplementedError(fmt::format("Not supported field:{} with message type", field_desc->name()));
        }
        break;
      }
    }
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status TableSchema::BuildFromFlatbuffers(const flatbuffers::TypeTable* type_table,
                                               const TableCreateOptions& opts) {
  for (size_t i = 0; i < type_table->num_elems; i++) {
    auto name = type_table->names[i];
    if (!opts.IsAllowed(name)) {
      continue;
    }
    absl::Status status;
    if (type_table->type_codes[i].is_repeating) {
      status = absl::UnimplementedError(fmt::format("Not supported field:{} with repeating base_type:{}", name,
                                                    static_cast<int>(type_table->type_codes[i].base_type)));
      break;
    }
    switch (type_table->type_codes[i].base_type) {
      case flatbuffers::ET_BOOL: {
        status = AddColumn<bool>(name);
        break;
      }
      case flatbuffers::ET_CHAR: {
        status = AddColumn<int8_t>(name);
        break;
      }
      case flatbuffers::ET_UCHAR: {
        status = AddColumn<uint8_t>(name);
        break;
      }
      case flatbuffers::ET_SHORT: {
        status = AddColumn<int16_t>(name);
        break;
      }
      case flatbuffers::ET_USHORT: {
        status = AddColumn<uint16_t>(name);
        break;
      }
      case flatbuffers::ET_INT: {
        status = AddColumn<int32_t>(name);
        break;
      }
      case flatbuffers::ET_UINT: {
        status = AddColumn<uint32_t>(name);
        break;
      }
      case flatbuffers::ET_LONG: {
        status = AddColumn<int64_t>(name);
        break;
      }
      case flatbuffers::ET_ULONG: {
        status = AddColumn<uint64_t>(name);
        break;
      }
      case flatbuffers::ET_FLOAT: {
        status = AddColumn<float>(name);
        break;
      }
      case flatbuffers::ET_DOUBLE: {
        status = AddColumn<double>(name);
        break;
      }
      case flatbuffers::ET_STRING: {
        status = AddColumn<std::string>(name);
        break;
      }
      default: {
        if (opts.ignore_unsupported_fields) {
          status = absl::OkStatus();
        } else {
          status = absl::UnimplementedError(fmt::format("Not supported field:{} with base_type:{}", name,
                                                        static_cast<int>(type_table->type_codes[i].base_type)));
        }
        break;
      }
    }
  }
  return absl::OkStatus();
}

void Table::Deleter::operator()(Table* ptr) {
  ptr->~Table();
  uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
  delete[] bytes;
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
}

absl::Status Table::BuildFromProtobufVector(const std::vector<const ::google::protobuf::Message*>& pb_vector) {
  const ::google::protobuf::Descriptor* desc = pb_vector[0]->GetDescriptor();
  const ::google::protobuf::Reflection* reflect = pb_vector[0]->GetReflection();

  for (int i = 0; i < desc->field_count(); i++) {
    const ::google::protobuf::FieldDescriptor* field_desc = desc->field(i);
    if (!schema_->ExistField(field_desc->name())) {
      continue;
    }
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
template <typename T>
absl::Status Table::SetColumnByFlatbuffersField(const std::vector<const uint8_t*>& fbs_vector, const std::string& name,
                                                size_t idx) {
  std::vector<T> vec;
  vec.reserve(fbs_vector.size());

  for (auto* fbs : fbs_vector) {
    const uint8_t* ptr = reinterpret_cast<const flatbuffers::Table*>(fbs)->GetAddressOf(
        flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(idx)));
    if (ptr == nullptr) {
      vec.emplace_back(T{});
      continue;
    }
    if constexpr (std::is_same_v<bool, T> || std::is_same_v<uint8_t, T> || std::is_same_v<int8_t, T> ||
                  std::is_same_v<uint16_t, T> || std::is_same_v<int16_t, T> || std::is_same_v<uint32_t, T> ||
                  std::is_same_v<int32_t, T> || std::is_same_v<int64_t, T> || std::is_same_v<uint64_t, T> ||
                  std::is_same_v<float, T> || std::is_same_v<double, T>) {
      vec.emplace_back(flatbuffers::ReadScalar<T>(ptr));
    } else if constexpr (std::is_same_v<StringView, T>) {
      ptr += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(ptr);
      const flatbuffers::String* str = reinterpret_cast<const flatbuffers::String*>(ptr);
      StringView s(reinterpret_cast<const char*>(str->Data()), str->size());
      vec.emplace_back(s);
    }
  }
  return Set(name, std::move(vec));
}

absl::Status Table::BuildFromFlatbuffersVector(const flatbuffers::TypeTable* type_table,
                                               const std::vector<const uint8_t*>& fbs_vector) {
  for (size_t i = 0; i < type_table->num_elems; i++) {
    auto name = type_table->names[i];
    if (!schema_->ExistField(name)) {
      continue;
    }
    absl::Status status;
    switch (type_table->type_codes[i].base_type) {
      case flatbuffers::ET_BOOL: {
        status = SetColumnByFlatbuffersField<bool>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_CHAR: {
        status = SetColumnByFlatbuffersField<int8_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_UCHAR: {
        status = SetColumnByFlatbuffersField<uint8_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_SHORT: {
        status = SetColumnByFlatbuffersField<int16_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_USHORT: {
        status = SetColumnByFlatbuffersField<uint16_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_INT: {
        status = SetColumnByFlatbuffersField<int32_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_UINT: {
        status = SetColumnByFlatbuffersField<uint32_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_LONG: {
        status = SetColumnByFlatbuffersField<int64_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_ULONG: {
        status = SetColumnByFlatbuffersField<uint64_t>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_FLOAT: {
        status = SetColumnByFlatbuffersField<float>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_DOUBLE: {
        status = SetColumnByFlatbuffersField<double>(fbs_vector, name, i);
        break;
      }
      case flatbuffers::ET_STRING: {
        status = SetColumnByFlatbuffersField<StringView>(fbs_vector, name, i);
        break;
      }
      default: {
        status = absl::UnimplementedError(fmt::format("Not supported field:{} with base_type:{}", name,
                                                      static_cast<int>(type_table->type_codes[i].base_type)));
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
  size_t count = Count();
  if (indices_.Size() == 0) {
    indices_ = functions::simd_vector_iota<int32_t>(ctx_, 0, count);
  }
  auto* p = ctx_.ArenaAllocate(sizeof(int32_t) * count);
  memcpy(p, indices_.Data(), sizeof(int32_t) * count);
  VectorData vdata(p, count, sizeof(int32_t) * indices_.Size());
  vdata.SetReadonly(false);
  return Vector<int32_t>(vdata);
}

void Table::SetColumn(uint32_t offset, VectorData vec) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  memcpy(vec_ptr, &vec, sizeof(vec));
}

size_t Table::Size() const { return schema_->FieldCount(); }

size_t Table::Count() const {
  auto* first_column = reinterpret_cast<const VectorData*>(this + 1);
  return first_column->Size();
}

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
  if (k >= Count()) {
    return this;
  }
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