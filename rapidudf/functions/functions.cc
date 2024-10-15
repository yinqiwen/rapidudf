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
#include "rapidudf/functions/functions.h"
#include <mutex>
#include <unordered_map>
#include "rapidudf/meta/function.h"
namespace rapidudf {
namespace functions {

static std::unordered_map<std::string, OpToken>& get_builtin_func_op_mapping() {
  static std::unordered_map<std::string, OpToken> mapping;
  return mapping;
}

extern void init_builtin_stl_sets_funcs();
extern void init_builtin_stl_maps_funcs();
extern void init_builtin_stl_vectors_funcs();
extern void init_builtin_strings_funcs();
extern void init_builtin_math_funcs();
extern void init_builtin_json_funcs();
extern void init_builtin_simd_vector_funcs();
extern void init_builtin_simd_table_funcs();

static std::once_flag g_init_builtin_flag;
void init_builtin() {
  for (uint32_t op = OP_UNARY_BEGIN; op < OP_END; op++) {
    get_builtin_func_op_mapping().emplace(kOpTokenStrs[op], static_cast<OpToken>(op));
  }
  std::call_once(g_init_builtin_flag, []() {
    init_builtin_math_funcs();
    init_builtin_stl_vectors_funcs();
    init_builtin_stl_maps_funcs();
    init_builtin_stl_sets_funcs();

    init_builtin_strings_funcs();
    init_builtin_json_funcs();
    init_builtin_simd_vector_funcs();
    init_builtin_simd_table_funcs();
  });
}

OpToken get_buitin_func_op(const std::string& name) {
  auto found = get_builtin_func_op_mapping().find(name);
  if (found != get_builtin_func_op_mapping().end()) {
    return found->second;
  }
  return OP_INVALID;
}

bool has_vector_buitin_func(OpToken op, DType dtype) {
  std::string fname = GetFunctionName(op, dtype.ToSimdVector());
  return FunctionFactory::GetFunction(fname) != nullptr;
}

}  // namespace functions
}  // namespace rapidudf