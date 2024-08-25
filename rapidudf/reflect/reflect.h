/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>
#include "absl/status/statusor.h"
#include "rapidudf/codegen/code_generator.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/value.h"
#include "xbyak/xbyak.h"

namespace rapidudf {

struct StructMember {
  std::optional<FunctionDesc> member_func;
  std::optional<DType> member_field_dtype;
  std::string field_name;
  uint32_t member_field_offset = 0;
  StructMember(const std::string& name, DType dtype, uint32_t offset) {
    field_name = name;
    member_field_dtype = dtype;
    member_field_offset = offset;
  }
  template <typename T, typename RET, typename... Args>
  StructMember(const std::string& name, RET (T::*f)(Args...)) {
    FunctionDesc desc;
    desc.name = name;
    desc.return_type = get_dtype<RET>();
    auto this_dtype = get_dtype<T>();
    this_dtype = this_dtype.ToPtr();
    desc.arg_types.emplace_back(this_dtype);
    (desc.arg_types.emplace_back(rapidudf::get_dtype<Args>()), ...);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpmf-conversions"
    desc.func = reinterpret_cast<void*>(f);
#pragma GCC diagnostic pop
    member_func = desc;
  }
  StructMember(const FunctionDesc& f) { member_func = f; }

  bool HasField() const { return member_field_dtype.has_value(); }
  bool HasMemberFunc() const { return member_func.has_value(); }

  absl::StatusOr<ValuePtr> BuildFuncCall(CodeGenerator& codegen, const Value& this_arg,
                                         const std::vector<ValuePtr>& args = {});
  absl::StatusOr<ValuePtr> BuildFieldAccess(CodeGenerator& codegen);
};
using StructMemberMap = std::unordered_map<std::string, StructMember>;
using GlobalStructMemberIndex = std::unordered_map<uint64_t, StructMemberMap>;

class ReflectFactory {
 public:
  static void Init();
  static std::optional<StructMember> GetStructMember(DType dtype, const std::string& name);

  static bool AddStructField(DType obj_dtype, const std::string& name, DType field_dtype, uint32_t field_offset);

  template <typename T, typename RET, typename... Args>
  static bool AddStructMethodAccessor(const std::string& name, RET (*f)(T*, Args...)) {
    // #pragma GCC diagnostic push
    // #pragma GCC diagnostic ignored "-Wpmf-conversions"
    void* ff = reinterpret_cast<void*>(f);
    // #pragma GCC diagnostic pop
    return AddStructMethod<T, RET, Args...>(name, ff);
  }

 private:
  static bool AddStructMethodAccessor(DType dtype, const std::string& name, const FunctionDesc& f);
  template <typename T, typename RET, typename... Args>
  static bool AddStructMethod(const std::string& name, void* f) {
    FunctionDesc desc;
    desc.name = name;
    desc.return_type = get_dtype<RET>();
    auto this_dtype = get_dtype<T>();
    this_dtype = this_dtype.ToPtr();
    desc.arg_types.emplace_back(this_dtype);
    (desc.arg_types.emplace_back(rapidudf::get_dtype<Args>()), ...);
    desc.func = reinterpret_cast<void*>(f);
    DTypeFactory::Add<T>();
    return AddStructMethodAccessor(get_dtype<T>(), name, desc);
  }
};
}  // namespace rapidudf