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
#include "rapidudf/types/dyn_object_schema.h"
#include <memory>
#include <mutex>
#include <utility>
#include "rapidudf/common/allign.h"
#include "rapidudf/log/log.h"
#include "rapidudf/types/dyn_object.h"

namespace rapidudf {

static constexpr uint32_t kDynObjHeaderAllign = 16;

using GlobalDynObjectSchemaTable =
    std::pair<std::mutex, absl::flat_hash_map<std::string, std::unique_ptr<DynObjectSchema>>>;
static GlobalDynObjectSchemaTable& get_schema_table() {
  static GlobalDynObjectSchemaTable table;
  return table;
}

DynObjectSchema* DynObjectSchema::GetMutable(const std::string& name) {
  auto& [table_mutex, table] = get_schema_table();
  std::lock_guard<std::mutex> guard(table_mutex);
  auto found = table.find(name);
  if (found != table.end()) {
    return found->second.get();
  }
  return nullptr;
}

const DynObjectSchema* DynObjectSchema::Get(const std::string& name) { return GetMutable(name); }

std::vector<std::string> DynObjectSchema::ListAll() {
  std::vector<std::string> names;
  auto& [table_mutex, table] = get_schema_table();
  std::lock_guard<std::mutex> guard(table_mutex);
  for (auto& [name, _] : table) {
    names.emplace_back(name);
  }
  return names;
}

DynObjectSchema::DynObjectSchema(const std::string& name, Options opts) : name_(name), opts_(opts) {
  allocated_offset_ = opts_.object_header_byte_size;
}

const DynObjectSchema* DynObjectSchema::GetOrCreate(const std::string& name, InitFunc&& init, Options opts) {
  return GetOrCreate(
      name, std::move(init), [](const std::string& name, Options opts) { return new DynObjectSchema(name, opts); },
      opts);
}

const DynObjectSchema* DynObjectSchema::GetOrCreate(const std::string& name, InitFunc&& init, NewFunc&& new_f,
                                                    Options opts) {
  if (opts.object_header_byte_size == 0) {
    opts.object_header_byte_size = sizeof(DynObject);
  }
  opts.object_header_byte_size = align_to<size_t>(opts.object_header_byte_size, 16);
  auto& [table_mutex, table] = get_schema_table();
  std::lock_guard<std::mutex> guard(table_mutex);
  auto found = table.find(name);
  if (found != table.end()) {
    return found->second.get();
  }

  std::unique_ptr<DynObjectSchema> schema(new_f(name, opts));
  DynObjectSchema* p = schema.get();
  init(p);
  table.emplace(name, std::move(schema));
  return p;
}

bool DynObjectSchema::ExistField(const std::string& name, const DType& dtype) const {
  auto found = fields_.find(name);
  if (found == fields_.end()) {
    return false;
  }
  return found->second.dtype == dtype;
}

absl::StatusOr<std::pair<DType, uint32_t>> DynObjectSchema::GetField(const std::string& name) const {
  auto found = fields_.find(name);
  if (found == fields_.end()) {
    return absl::NotFoundError(fmt::format("no field:{}", name));
  }
  return std::make_pair(found->second.dtype, found->second.bytes_offset);
}
absl::StatusOr<reflect::Field> DynObjectSchema::Add(const std::string& name, DType dtype) {
  if (dtype.IsPrimitive() || dtype.IsSimdVectorPtr()) {
    // allowed dtype
    uint32_t byte_size = dtype.ByteSize();
    if (byte_size < 8) {
      byte_size = 8;
    }
    reflect::Field field;
    field.bytes_offset = allocated_offset_;
    field.dtype = dtype;

    auto [iter, success] = fields_.emplace(name, field);
    if (!success) {
      RUDF_RETURN_FMT_ERROR("Duplicate field name:{} to add into DynObject:{}", name, name_);
    }
    allocated_offset_ += byte_size;
    return field;
  } else {
    RUDF_RETURN_FMT_ERROR("Unsupported dtype:{}", dtype);
  }
}

typename DynObject::SmartPtr DynObjectSchema::NewObject() const {
  uint8_t* bytes = new uint8_t[ByteSize()];
  memset(bytes, 0, ByteSize());
  try {
    new (bytes) DynObject(this);
  } catch (...) {
    throw;
  }
  DynObject::SmartPtr p(reinterpret_cast<DynObject*>(bytes));
  return p;
}

void DynObjectSchema::VisitField(std::function<void(const std::string&, const DType&, uint32_t)>&& f) const {
  for (auto& [name, field] : fields_) {
    f(name, field.dtype, field.bytes_offset);
  }
}

std::string DynObjectSchema::ToString() const {
  std::string info;
  VisitField([&](const std::string& name, const DType& dtype, uint32_t) {
    if (!info.empty()) {
      info.append(",");
    }
    info.append("{field:").append(name).append(", dtype:").append(dtype.GetTypeString()).append("}");
  });
  return "[" + info + "]";
}

}  // namespace rapidudf