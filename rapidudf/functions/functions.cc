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