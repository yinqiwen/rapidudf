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
#include "rapidudf/vector/table_schema.h"
#include <memory>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/types/bit.h"
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
  if (!s->IsTable()) {
    RUDF_ERROR("DynObject:{}  is not table.", name);
    return nullptr;
  }
  return reinterpret_cast<TableSchema*>(s);
}

const TableSchema* TableSchema::Get(const std::string& name) { return GetMutable(name); }

const reflect::Column* TableSchema::GetColumnByIdx(uint32_t idx) const {
  if (idx >= columns_.size()) {
    return nullptr;
  }
  return &columns_[idx];
}

bool TableSchema::ExistRow(const RowSchema& row) const {
  for (auto& s : row_schemas_) {
    if (*s == row) {
      return true;
    }
  }
  return false;
}

absl::Status TableSchema::AddColumn(const std::string& name, const DType& dtype, const RowSchema* schema,
                                    uint32_t field_idx) {
  auto result = Add(name, dtype);
  if (!result.ok()) {
    return result.status();
  }
  reflect::Column column;
  column.name = name;
  column.schema = schema;
  column.field = result.value();
  column.field_idx = field_idx;
  columns_.emplace_back(std::move(column));
  return absl::OkStatus();
}

bool TableSchema::ExistColumn(const std::string& name, const DType& dtype) const {
  return DynObjectSchema::ExistField(name, dtype);
}

const TableSchema* TableSchema::GetOrCreate(const std::string& name, InitFunc&& init,
                                            const TableCreateOptions& table_opts) {
  Options opts;
  opts.object_header_byte_size = sizeof(Table);
  opts.is_table = true;
  const DynObjectSchema* s = DynObjectSchema::GetOrCreate(
      name,
      [&](DynObjectSchema* s) {
        TableSchema* table_schema = reinterpret_cast<TableSchema*>(s);
        init(table_schema);
      },
      [table_opts](const std::string& name, Options opts) { return new TableSchema(name, opts, table_opts); }, opts);
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

absl::Status TableSchema::AddColumns(const ::google::protobuf::Message* msg) {
  const ::google::protobuf::Descriptor* desc = msg->GetDescriptor();
  RowSchemaPtr schema = std::make_unique<RowSchema>(desc);
  uint32_t valid_column = 0;
  for (int i = 0; i < desc->field_count(); i++) {
    const ::google::protobuf::FieldDescriptor* field_desc = desc->field(i);
    if (!table_opts_.IsAllowed(field_desc->name())) {
      continue;
    }
    absl::Status status;
    switch (field_desc->type()) {
      case ::google::protobuf::FieldDescriptor::TYPE_BOOL: {
        status = AddColumn<bool>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_DOUBLE: {
        status = AddColumn<double>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FLOAT: {
        status = AddColumn<float>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SINT64:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED64:
      case ::google::protobuf::FieldDescriptor::TYPE_INT64: {
        status = AddColumn<int64_t>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED64:
      case ::google::protobuf::FieldDescriptor::TYPE_UINT64: {
        status = AddColumn<uint64_t>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_SINT32:
      case ::google::protobuf::FieldDescriptor::TYPE_SFIXED32:
      case ::google::protobuf::FieldDescriptor::TYPE_INT32: {
        status = AddColumn<int32_t>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_FIXED32:
      case ::google::protobuf::FieldDescriptor::TYPE_UINT32: {
        status = AddColumn<uint32_t>(field_desc->name(), schema.get(), i);
        break;
      }
      case ::google::protobuf::FieldDescriptor::TYPE_STRING:
      case ::google::protobuf::FieldDescriptor::TYPE_BYTES: {
        status = AddColumn<std::string>(field_desc->name(), schema.get(), i);
        break;
      }
      default: {
        if (table_opts_.ignore_unsupported_fields) {
          continue;
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
    valid_column++;
  }
  if (valid_column == 0) {
    RUDF_LOG_RETURN_FMT_ERROR("No valid column found in pb:{}", msg->GetTypeName());
  }
  row_schemas_.emplace_back(std::move(schema));
  return absl::OkStatus();
}

absl::Status TableSchema::AddColumns(const flatbuffers::TypeTable* type_table) {
  RowSchemaPtr schema = std::make_unique<RowSchema>(type_table);
  uint32_t valid_column = 0;
  for (size_t i = 0; i < type_table->num_elems; i++) {
    auto name = type_table->names[i];
    if (!table_opts_.IsAllowed(name)) {
      continue;
    }
    absl::Status status;
#if FLATBUFFERS_VERSION_MAJOR == 1
    if (type_table->type_codes[i].is_vector) {
#else
    if (type_table->type_codes[i].is_repeating) {
#endif

      if (table_opts_.ignore_unsupported_fields) {
        continue;
      }
      status = absl::UnimplementedError(fmt::format("Not supported field:{} with repeating base_type:{}", name,
                                                    static_cast<int>(type_table->type_codes[i].base_type)));
      return status;
    }
    switch (type_table->type_codes[i].base_type) {
      case flatbuffers::ET_BOOL: {
        status = AddColumn<bool>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_CHAR: {
        status = AddColumn<int8_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_UCHAR: {
        status = AddColumn<uint8_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_SHORT: {
        status = AddColumn<int16_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_USHORT: {
        status = AddColumn<uint16_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_INT: {
        status = AddColumn<int32_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_UINT: {
        status = AddColumn<uint32_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_LONG: {
        status = AddColumn<int64_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_ULONG: {
        status = AddColumn<uint64_t>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_FLOAT: {
        status = AddColumn<float>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_DOUBLE: {
        status = AddColumn<double>(name, schema.get(), i);
        break;
      }
      case flatbuffers::ET_STRING: {
        status = AddColumn<std::string>(name, schema.get(), i);
        break;
      }
      default: {
        if (table_opts_.ignore_unsupported_fields) {
          continue;
        } else {
          status = absl::UnimplementedError(fmt::format("Not supported field:{} with base_type:{}", name,
                                                        static_cast<int>(type_table->type_codes[i].base_type)));
        }
        break;
      }
    }
    if (!status.ok()) {
      return status;
    }
    valid_column++;
  }
  if (valid_column == 0) {
    RUDF_LOG_RETURN_FMT_ERROR("No valid column found in fbs");
  }
  row_schemas_.emplace_back(std::move(schema));
  return absl::OkStatus();
}

absl::Status TableSchema::AddColumns(const DType& dtype) {
  auto members = Reflect::GetStructMembers(dtype);
  if (members == nullptr) {
    RUDF_LOG_RETURN_FMT_ERROR("Unsupported dtype:{}", dtype);
  }
  RowSchemaPtr schema = std::make_unique<RowSchema>(dtype);
  uint32_t valid_column = 0;
  for (size_t i = 0; i < members->size(); i++) {
    auto* member = members->at(i);
    auto name = member->name;
    DType member_dtype;

    if (!member->HasField()) {
      if (member->member_func->arg_types.size() == 1 && member->member_func->arg_types[0] == dtype.ToPtr()) {
        member_dtype = member->member_func->return_type;
      } else {
        continue;
      }
    } else {
      member_dtype = (*member->member_field_dtype).Elem();
    }
    if (!table_opts_.IsAllowed(name)) {
      continue;
    }

    bool valid_dtype = false;
    absl::Status status;
    if (!member_dtype.IsFundamental() || !member_dtype.IsPrimitive()) {
      if (member_dtype.IsString() || member_dtype.IsStringPtr()) {
        valid_dtype = true;
      } else {
        valid_dtype = false;
      }
    } else {
      valid_dtype = true;
    }
    if (!valid_dtype) {
      if (table_opts_.ignore_unsupported_fields) {
        continue;
      }
      RUDF_LOG_RETURN_FMT_ERROR("Not supported field:{} with dtype:{}", name, member_dtype);
    }

    switch (member_dtype.GetFundamentalType()) {
      case DATA_I8: {
        status = AddColumn<int8_t>(name, schema.get(), i);
        break;
      }
      case DATA_U8: {
        status = AddColumn<uint8_t>(name, schema.get(), i);
        break;
      }
      case DATA_I16: {
        status = AddColumn<int16_t>(name, schema.get(), i);
        break;
      }
      case DATA_U16: {
        status = AddColumn<uint16_t>(name, schema.get(), i);
        break;
      }
      case DATA_I32: {
        status = AddColumn<int32_t>(name, schema.get(), i);
        break;
      }
      case DATA_U32: {
        status = AddColumn<uint32_t>(name, schema.get(), i);
        break;
      }
      case DATA_I64: {
        status = AddColumn<int64_t>(name, schema.get(), i);
        break;
      }
      case DATA_U64: {
        status = AddColumn<uint64_t>(name, schema.get(), i);
        break;
      }
      case DATA_F32: {
        status = AddColumn<float>(name, schema.get(), i);
        break;
      }
      case DATA_F64: {
        status = AddColumn<double>(name, schema.get(), i);
        break;
      }
      case DATA_STRING:
      case DATA_STD_STRING_VIEW:
      case DATA_STRING_VIEW: {
        status = AddColumn<std::string>(name, schema.get(), i);
        break;
      }
      case DATA_BIT: {
        status = AddColumn<bool>(name, schema.get(), i);
        break;
      }
      default: {
        if (table_opts_.ignore_unsupported_fields) {
          continue;
        } else {
          RUDF_LOG_RETURN_FMT_ERROR("Not supported field:{} with dtype:{}", name, member_dtype);
        }
        break;
      }
    }

    if (!status.ok()) {
      return status;
    }
    valid_column++;
  }
  if (valid_column == 0) {
    RUDF_LOG_RETURN_FMT_ERROR("No valid column found in struct:{} with members:{}", dtype, members->size());
  }
  row_schemas_.emplace_back(std::move(schema));
  return absl::OkStatus();
}
}  // namespace simd
}  // namespace rapidudf