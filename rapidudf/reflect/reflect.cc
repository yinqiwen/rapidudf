/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/reflect/reflect.h"
#include <vector>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/stl_access.h"
namespace rapidudf {
using namespace Xbyak::util;

static void init_reflect_builtin() {
  static bool inited = false;
  if (inited) {
    return;
  }
  inited = true;
  init_stl_reflect_access();
}

static GlobalStructMemberIndex& get_global_reflect_index() {
  static GlobalStructMemberIndex index;
  init_reflect_builtin();
  return index;
}
void ReflectFactory::Init() { init_reflect_builtin(); }
std::optional<StructMember> ReflectFactory::GetStructMember(DType dtype, const std::string& name) {
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
bool ReflectFactory::AddStructField(DType obj_dtype, const std::string& name, DType field_dtype,
                                    uint32_t field_offset) {
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
bool ReflectFactory::AddStructMethodAccessor(DType dtype, const std::string& name, const FuncDesc& f) {
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

absl::StatusOr<ValuePtr> StructMember::BuildFieldAccess(CodeGenerator& codegen) {
  if (!member_field_dtype.has_value()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("member_field_dtype is null to build field access"));
  }

  uint32_t bits = member_field_dtype->Bits();
  DType val_dtype = *member_field_dtype;
  RUDF_DEBUG("access field dtype:{}", *member_field_dtype);
  codegen.GetCodeGen().add(rcx, member_field_offset);
  if (member_field_dtype->IsNumber()) {
    // return number value
    codegen.GetCodeGen().mov(rax.changeBit(bits), ptr[rcx]);
    return codegen.NewValueByRegister(val_dtype, rax);
  } else if (member_field_dtype->IsPtr()) {
    // return ptr value
    codegen.GetCodeGen().mov(rax, ptr[rcx]);
    return codegen.NewValueByRegister(val_dtype, rax);
  } else if (member_field_dtype->IsAbslSpan()) {
    codegen.GetCodeGen().mov(rax, ptr[rcx]);
    codegen.GetCodeGen().add(rcx, 8);
    codegen.GetCodeGen().mov(rdx, ptr[rcx]);
    return codegen.NewValueByRegister(val_dtype, {&rax, &rdx});

  } else if (member_field_dtype->IsStringView()) {
    codegen.GetCodeGen().mov(rax, ptr[rcx]);
    codegen.GetCodeGen().add(rcx, 8);
    codegen.GetCodeGen().mov(rdx, ptr[rcx]);
    return codegen.NewValueByRegister(val_dtype, {&rax, &rdx});
    // typedef void (*string_view_data_access)(std::string_view*);
    // typedef void (*string_view_size_access)(std::string_view*);
    // string_view_data_access data_ptr = reinterpret_cast<string_view_data_access>(&std::string_view::data);
    // string_view_data_access size_ptr = reinterpret_cast<string_view_data_access>(&std::string_view::size);
    // c.mov(rax, (size_t)data_ptr);
    // c.call(rax);
    // c.mov(rdx, rax);
    // c.mov(rax, (size_t)size_ptr);
    // c.call(rax);
  } else {
    // return ptr for normal object
    codegen.GetCodeGen().mov(rax, rcx);
    val_dtype = val_dtype.ToPtr();
    return codegen.NewValueByRegister(val_dtype, rax);
  }
}

absl::StatusOr<ValuePtr> StructMember::BuildFuncCall(CodeGenerator& codegen, const Value& this_arg,
                                                     const std::vector<ValuePtr>& args) {
  if (!member_func.has_value()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError("member_func is null to build func call"));
  }
  if ((args.size() + 1) != member_func->arg_types.size()) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("Can NOT invoke object func:{} with {} args, which need {} args.",
                                               member_func->name, member_func->arg_types.size() - 1, args.size())));
  }
  auto func_arg_regs = GetFuncArgsRegistersByDTypes(member_func->arg_types);
  if (func_arg_regs.size() != member_func->arg_types.size()) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("Can NOT allocate registers for object func:{}'s  {} args.",
                                               member_func->name, member_func->arg_types.size())));
  }
  // Value this_arg(&codegen, this_dtype, &rcx, false);
  std::vector<const Value*> func_all_args{&this_arg};
  for (auto p : args) {
    func_all_args.emplace_back(p.get());
  }
  ValuePtr result = codegen.CallFunction(*member_func, func_all_args);
  if (!result) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("Can NOT invoke object func:{} with nil return value")));
  }
  return result;
}

}  // namespace rapidudf