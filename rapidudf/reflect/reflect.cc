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
#include "rapidudf/reflect/reflect.h"
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {

static GlobalStructMemberIndex& get_global_reflect_index() {
  static GlobalStructMemberIndex index;
  return index;
}

std::optional<StructMember> Reflect::GetStructMember(DType dtype, const std::string& name) {
  auto found = get_global_reflect_index().find(dtype.Control());
  if (found == get_global_reflect_index().end()) {
    return {};
  }
  auto iter = found->second.find(name);
  if (iter == found->second.end()) {
    return {};
  }
  return iter->second;
}
bool Reflect::AddStructField(DType obj_dtype, const std::string& name, DType field_dtype, uint32_t field_offset) {
  auto [iter, success] =
      get_global_reflect_index()[obj_dtype.Control()].emplace(name, StructMember(name, field_dtype, field_offset));
  if (!success) {
    StructMember& entry = iter->second;
    if (!entry.member_field_dtype.has_value()) {
      entry.member_field_dtype = field_dtype;
      entry.member_field_offset = field_offset;
      entry.field_name = name;
      return true;
    }
  }
  return success;
}
bool Reflect::AddStructMethodAccessor(DType dtype, const std::string& name, const FunctionDesc& f) {
  auto [iter, success] = get_global_reflect_index()[dtype.Control()].emplace(name, StructMember(f));
  if (!success) {
    StructMember& entry = iter->second;
    if (!entry.member_func.has_value()) {
      entry.member_func = f;
      return true;
    }
  }
  return success;
}

}  // namespace rapidudf