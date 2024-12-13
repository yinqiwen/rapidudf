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

#pragma once
#include <string>
#include "rapidudf/reflect/struct.h"
#include "rapidudf/table/row.h"

namespace rapidudf {
namespace table {
struct Column {
  std::string name;
  reflect::Field field;
  const RowSchema* schema = nullptr;
  uint32_t field_idx = 0;

  const ::google::protobuf::FieldDescriptor* GetProtobufField() const { return schema->pb_desc->field(field_idx); }
  const StructMember* GetStructField() const { return schema->struct_members->at(field_idx); }
};
}  // namespace table
}  // namespace rapidudf