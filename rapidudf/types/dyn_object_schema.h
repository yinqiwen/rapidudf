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

#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/dyn_object.h"

namespace rapidudf {
class DynObjectSchema {
 public:
  struct Options {
    size_t object_header_byte_size;
    size_t object_body_byte_size;
    bool is_table;
    Options() {
      object_header_byte_size = 0;
      object_body_byte_size = 0;
      is_table = false;
    }
  };

  using InitFunc = std::function<void(DynObjectSchema* s)>;
  static const DynObjectSchema* GetOrCreate(const std::string& name, InitFunc&& init, Options opts = Options());
  static const DynObjectSchema* Get(const std::string& name);
  static DynObjectSchema* GetMutable(const std::string& name);
  static std::vector<std::string> ListAll();

  typename DynObject::SmartPtr NewObject() const;
  bool ExistField(const std::string& name) const;

  template <typename T>
  absl::Status AddField(const std::string& name) {
    return Add(name, get_dtype<T>());
  }

  absl::StatusOr<std::pair<DType, uint32_t>> GetField(const std::string& name) const;

  uint32_t ByteSize() const { return allocated_offset_; }

  bool IsTable() const { return opts_.is_table; }

  void VisitField(std::function<void(const std::string&, const DType&, uint32_t)>&& f) const;

  size_t FieldCount() const { return fields_.size(); }

 protected:
  DynObjectSchema(const std::string& name, Options opts);
  // void SetFlags(Flags flags) { flags_ = flags; }

  absl::Status Add(const std::string& name, DType dtype);

  struct Field {
    DType dtype;
    uint32_t bytes_offset = 0;
  };

  using FieldTable = absl::flat_hash_map<std::string, Field>;
  std::string name_;
  Options opts_;
  FieldTable fields_;
  uint32_t allocated_offset_ = 0;
};
using DynObjectSchemaMap = std::map<std::string, DynObjectSchema*>;
}  // namespace rapidudf