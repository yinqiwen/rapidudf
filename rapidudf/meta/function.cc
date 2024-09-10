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
#include "rapidudf/meta/function.h"
namespace rapidudf {
using FuncRegMap = std::unordered_map<std::string, FunctionDesc>;
static std::unique_ptr<FuncRegMap> g_regs = nullptr;

static inline uint32_t allgin_n(uint32_t x, uint32_t n) { return (x + n - 1) & ~(n - 1); }

static std::string get_simd_vector_func_suffix(const std::vector<DType>& dtypes) {
  std::string suffix;
  for (auto dtype : dtypes) {
    if (dtype.IsSimdVector()) {
      suffix.append("_vector");
    } else {
      suffix.append("_scalar");
    }
  }
  return suffix;
}

std::string GetFunctionName(std::string_view op, DType dtype) {
  std::string fname(op);
  fname = fname + "_" + dtype.Elem().GetTypeString();
  if (dtype.IsSimdVector()) {
    return std::string(FunctionFactory::kSimdVectorUnaryFuncPrefix) + "_" + fname;
  } else {
    return fname;
  }
}
std::string GetFunctionName(std::string_view op, DType a, DType b) {
  std::string fname(op);
  std::string arg_types;
  if (a.IsSimdVector() && b.IsSimdVector()) {
    if (a.Elem() != b.Elem()) {
      arg_types = a.Elem().GetTypeString() + "_" + b.Elem().GetTypeString();
    } else {
      arg_types = a.Elem().GetTypeString();
    }
  } else if (a.IsSimdVector()) {
    arg_types = a.Elem().GetTypeString();
  } else if (b.IsSimdVector()) {
    arg_types = b.Elem().GetTypeString();
  } else {
    arg_types = b.Elem().GetTypeString();
  }
  fname = fname + "_" + arg_types;
  if (a.IsSimdVector() || b.IsSimdVector()) {
    fname = std::string(FunctionFactory::kSimdVectorBinaryFuncPrefix) + "_" + fname;
    return fname + get_simd_vector_func_suffix({a, b});
  } else {
    return fname;
  }
}
std::string GetFunctionName(std::string_view op, DType a, DType b, DType c) {
  std::string fname(op);
  DType ele_type;
  if (c.IsSimdVector()) {
    ele_type = c.Elem();
  } else if (b.IsSimdVector()) {
    ele_type = b.Elem();
  } else {
    ele_type = a.Elem();
    if (op == kOpTokenStrs[OP_CONDITIONAL]) {
      ele_type = b;
    }
  }
  fname = fname + "_" + ele_type.GetTypeString();
  if (a.IsSimdVector() || b.IsSimdVector() || c.IsSimdVector()) {
    fname = std::string(FunctionFactory::kSimdVectorTernaryFuncPrefix) + "_" + fname;
    return fname + get_simd_vector_func_suffix({a, b, c});
  } else {
    return fname;
  }
}

std::string GetFunctionName(std::string_view op, const std::vector<DType>& arg_dtypes) {
  if (arg_dtypes.size() == 1) {
    return GetFunctionName(op, arg_dtypes[0]);
  } else if (arg_dtypes.size() == 2) {
    return GetFunctionName(op, arg_dtypes[0], arg_dtypes[1]);
  } else if (arg_dtypes.size() == 3) {
    return GetFunctionName(op, arg_dtypes[0], arg_dtypes[1], arg_dtypes[2]);
  } else {
    return std::string(op);
  }
}

std::string GetFunctionName(OpToken op, DType dtype) { return GetFunctionName(kOpTokenStrs[op], dtype); }
std::string GetFunctionName(OpToken op, DType a, DType b) { return GetFunctionName(kOpTokenStrs[op], a, b); }
std::string GetFunctionName(OpToken op, DType a, DType b, DType c) {
  return GetFunctionName(kOpTokenStrs[op], a, b, c);
}

bool FunctionDesc::ValidateArgs(const std::vector<DType>& ts) const {
  if (arg_types.size() != ts.size()) {
    return false;
  }
  for (size_t i = 0; i < ts.size(); i++) {
    if (!ts[i].CanCastTo(arg_types[i])) {
      return false;
    }
  }
  return true;
}

bool FunctionFactory::Register(FunctionDesc&& desc) {
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