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
#include <optional>
#include <unordered_map>
#include <vector>
#include "absl/status/statusor.h"
// #include "rapidudf/codegen/code_generator.h"
#include "rapidudf/meta/function.h"
// #include "rapidudf/codegen/value.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
template <uint64_t SOURCE, uint32_t LINE, uint64_t HASH, typename T>
struct ReflectRegisterHelper {};

struct StructMember {
  std::optional<FunctionDesc> member_func;
  std::optional<DType> member_field_dtype;
  std::string field_name;
  uint32_t member_field_offset = 0;
  StructMember() = default;
  explicit StructMember(const std::string& name, DType dtype, uint32_t offset) {
    field_name = name;
    member_field_dtype = dtype;
    member_field_offset = offset;
  }
  explicit StructMember(const FunctionDesc& f) { member_func = f; }

  bool HasField() const { return member_field_dtype.has_value(); }
  bool HasMemberFunc() const { return member_func.has_value(); }
};
using StructMemberMap = std::unordered_map<std::string, StructMember>;
using GlobalStructMemberIndex = std::unordered_map<uint64_t, StructMemberMap>;

class Reflect {
 public:
  static std::optional<StructMember> GetStructMember(DType dtype, const std::string& name);

  static bool AddStructField(DType obj_dtype, const std::string& name, DType field_dtype, uint32_t field_offset);

  template <typename T, typename RET, typename... Args>
  static bool AddStructMethodAccessor(const std::string& name, RET (*f)(T*, Args...)) {
    void* ff = reinterpret_cast<void*>(f);
    return AddStructMethod<T, RET, Args...>(name, ff, true);
  }
  template <typename T, typename RET, typename... Args>
  static bool AddStructMethodAccessor(const std::string& name, RET (*f)(T, Args...)) {
    void* ff = reinterpret_cast<void*>(f);
    return AddStructMethod<T, RET, Args...>(name, ff, false);
  }

 private:
  static bool AddStructMethodAccessor(DType dtype, const std::string& name, const FunctionDesc& f);
  template <typename T, typename RET, typename... Args>
  static bool AddStructMethod(const std::string& name, void* f, bool ptr) {
    FunctionDesc desc;
    desc.name = name;
    desc.return_type = get_dtype<RET>();
    auto this_dtype = get_dtype<T>();
    if (ptr) {
      this_dtype = this_dtype.ToPtr();
    }
    desc.arg_types.emplace_back(this_dtype);
    (desc.arg_types.emplace_back(rapidudf::get_dtype<Args>()), ...);
    desc.func = reinterpret_cast<void*>(f);
    DTypeFactory::Add<T>();
    return AddStructMethodAccessor(get_dtype<T>(), name, desc);
  }
};
}  // namespace rapidudf