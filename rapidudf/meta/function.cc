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
#include "rapidudf/meta/function.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
using FuncRegMap = std::unordered_map<std::string, FunctionDesc>;
static std::unique_ptr<FuncRegMap> g_regs = nullptr;

// static inline uint32_t allgin_n(uint32_t x, uint32_t n) { return (x + n - 1) & ~(n - 1); }

std::string GetFunctionName(std::string_view op, DType dtype) {
  std::string fname(op);
  if (dtype.IsSimdVector() || dtype.IsSimdVectorPtr()) {
    fname = fname + "_" + dtype.Elem().GetTypeString();
    return std::string(FunctionFactory::kSimdVectorFuncPrefix) + "_" + fname;
  } else {
    fname = fname + "_" + dtype.Elem().GetTypeString();
    return fname;
  }
}
std::string GetFunctionName(std::string_view op, DType dtype0, DType dtype1) {
  std::string fname(op);
  if (dtype0.IsSimdVector() || dtype1.IsSimdVector() || dtype0.IsSimdVectorPtr() || dtype1.IsSimdVectorPtr()) {
    fname = fname + "_" + dtype0.Elem().GetTypeString() + "_" + dtype1.Elem().GetTypeString();
    return std::string(FunctionFactory::kSimdVectorFuncPrefix) + "_" + fname;
  } else {
    fname = fname + "_" + dtype0.Elem().GetTypeString() + "_" + dtype1.Elem().GetTypeString();
    return fname;
  }
}

std::string GetMemberFuncName(DType dtype, const std::string& member) {
  std::string fname = fmt::format("{}_{}", dtype.GetTypeString(), member);
  return fname;
}

std::string GetFunctionName(OpToken op, DType dtype) { return GetFunctionName(kOpTokenStrs[op], dtype); }
std::string GetFunctionName(OpToken op, DType dtype0, DType dtype1) {
  return GetFunctionName(kOpTokenStrs[op], dtype0, dtype1);
}

void FunctionDesc::Init() {
  for (size_t i = 0; i < arg_types.size(); i++) {
    if (arg_types[i].IsContextPtr()) {
      if (context_arg_idx != -1) {
        RUDF_ERROR("Function:{} has more than ONE arg type is ContextPtr({}), the first is at:{}", i, context_arg_idx);
      } else {
        context_arg_idx = static_cast<int>(i);
      }
    }
  }
}
uint32_t FunctionDesc::GetOperandCount() const {
  uint32_t n = arg_types.size();
  if (context_arg_idx >= 0) {
    n--;
  }
  if (is_vector_func) {
    n--;
  }
  return n;
}

const DType& FunctionDesc::LastArg() const { return arg_types[arg_types.size() - 1]; }
bool FunctionDesc::PassArgByValue(size_t argno) const {
  if (argno >= arg_types.size()) {
    return false;
  }
  // x86-64 linux
  uint32_t used_param_registers = 0;
  uint32_t total_param_registers = 6;
  for (size_t i = 0; i <= argno; i++) {
    uint32_t request_param_registers = 0;
    if (arg_types[i].IsPtr() || arg_types[i].IsInteger() || arg_types[i].IsBit()) {
      request_param_registers = 1;
    } else if (arg_types[i].IsAbslSpan() || arg_types[i].IsStringView() || arg_types[i].IsStdStringView()) {
      request_param_registers = 2;
    }
    used_param_registers += request_param_registers;
  }
  if (arg_types[argno].IsAbslSpan() || arg_types[argno].IsStringView() || arg_types[argno].IsStdStringView()) {
    if (used_param_registers > total_param_registers) {
      return true;
    }
  }
  return false;
}

bool FunctionDesc::ValidateArgs(const std::vector<DType>& ts) const {
  size_t verify_arg_size = arg_types.size();
  if (is_vector_func) {
    verify_arg_size--;
  }
  if (ts.size() != verify_arg_size) {
    RUDF_ERROR("Func:{} expect {} args, while only {} args given", verify_arg_size, ts.size());
    return false;
  }
  for (size_t i = 0; i < verify_arg_size; i++) {
    if (is_vector_func) {
      if (ts[i].IsSimdVectorPtr()) {
        if (ts[i].Elem() != arg_types[i].PtrTo()) {
          RUDF_ERROR("Func:{} arg[{}] dtype:{}, while given arg value dtype:{}", name, i, arg_types[i], ts[i]);
          return false;
        }
      } else {
        if (!ts[i].CanCastTo(arg_types[i].PtrTo())) {
          RUDF_ERROR("Func:{} arg[{}] dtype:{}, while given arg value dtype:{}", name, i, arg_types[i], ts[i]);
          return false;
        }
      }
    } else {
      if (!ts[i].CanCastTo(arg_types[i])) {
        RUDF_ERROR("Func:{} arg[{}] dtype:{}, while given arg value dtype:{}", name, i, arg_types[i], ts[i]);
        return false;
      }
    }
  }
  return true;
}

bool FunctionDesc::CompareSignature(DType rtype, const std::vector<DType>& validate_args_types) const {
  if (return_type != rtype) {
    return false;
  }
  if (arg_types.size() != validate_args_types.size()) {
    return false;
  }
  for (size_t i = 0; i < arg_types.size(); i++) {
    if (arg_types[i] != validate_args_types[i]) {
      return false;
    }
  }
  return true;
}

bool FunctionFactory::Register(FunctionDesc&& desc) {
  desc.Init();
  if (!g_regs) {
    g_regs = std::make_unique<FuncRegMap>();
  }
  if (g_regs->count(desc.name) > 0) {
    RUDF_CRITICAL("Duplicate func name:{}", desc.name);
    return false;
  }
  // RUDF_DEBUG("Registe function:{}", desc.name);
  return g_regs->emplace(desc.name, desc).second;
}
absl::Status FunctionFactory::RegisterVectorFunction(FunctionDesc&& desc) {
  if (desc.arg_types.size() < 2) {
    RUDF_LOG_RETURN_FMT_ERROR("Vector function MUST have more than 2 args, while func:{} has {} args", desc.name,
                              desc.arg_types.size());
  }
  // const DType& first_arg_dtype = desc.arg_types[0];
  // const DType& last_arg_dtype = desc.arg_types[desc.arg_types.size() - 1];
  // if (first_arg_dtype != last_arg_dtype) {
  //   RUDF_LOG_RETURN_FMT_ERROR("Vector function:{}'s first and last arg have different type:{}/{}", desc.name,
  //                             first_arg_dtype, last_arg_dtype);
  // }
  for (auto& arg_dtype : desc.arg_types) {
    if (!arg_dtype.IsPtr()) {
      RUDF_LOG_RETURN_FMT_ERROR("Vector function:{}'s  arg is not all pointer", desc.name);
    }
  }
  desc.is_vector_func = true;
  if (!Register(std::move(desc))) {
    RUDF_LOG_RETURN_FMT_ERROR("Vector function:{} register faield.", desc.name);
  }
  return absl::OkStatus();
}

const FunctionDesc* FunctionFactory::GetFunction(const std::string& name) {
  if (!g_regs) {
    return nullptr;
  }
  auto found = g_regs->find(name);
  if (found == g_regs->end()) {
    return nullptr;
  }
  return &(found->second);
}
}  // namespace rapidudf