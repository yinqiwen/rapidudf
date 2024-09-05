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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/simd/simd_ops.h"
#include "rapidudf/types/simd.h"
namespace rapidudf {
#define REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();
#define REGISTER_SIMD_VECTOR_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

template <typename T, OpToken op>
static void register_simd_vector_arithmetic() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f);
  simd_vector_func_name = fmt::format("simd_vector_scalar_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_ff)(simd::Vector<T>, T, bool, uint32_t) = simd::simd_binary_scalar_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_ff);
}

template <typename T, OpToken op>
static void register_simd_vector_cmp() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<Bit> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, Bit, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f);
  simd_vector_func_name = fmt::format("simd_vector_scalar_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<Bit> (*simd_ff)(simd::Vector<T>, T, bool, uint32_t) = simd::simd_binary_scalar_op<T, Bit, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_ff);
}

template <typename T>
static void register_simd_vector_ternary() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_ternary_{}_scalar_scalar", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f0)(simd::Vector<Bit>, T, T, uint32_t) = simd::simd_ternary_op_scalar_scalar<T>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f0);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_vector_vector", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f1)(simd::Vector<Bit>, simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_ternary_op_vector_vector<T>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f1);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_vector_scalar", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f2)(simd::Vector<Bit>, simd::Vector<T>, T, uint32_t) = simd::simd_ternary_op_vector_scalar<T>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f2);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_scalar_vector", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f3)(simd::Vector<Bit>, T, simd::Vector<T>, uint32_t) = simd::simd_ternary_op_scalar_vector<T>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f3);
}

template <typename T>
static void register_simd_vector_add() {
  register_simd_vector_arithmetic<T, OP_PLUS>();
}
template <typename T>
static void register_simd_vector_sub() {
  register_simd_vector_arithmetic<T, OP_MINUS>();
}
template <typename T>
static void register_simd_vector_mul() {
  register_simd_vector_arithmetic<T, OP_MULTIPLY>();
}
template <typename T>
static void register_simd_vector_div() {
  register_simd_vector_arithmetic<T, OP_DIVIDE>();
}
template <typename T>
static void register_simd_vector_mod() {
  register_simd_vector_arithmetic<T, OP_MOD>();
}

template <typename T>
static void register_simd_vector_gt() {
  register_simd_vector_cmp<T, OP_GREATER>();
}

template <typename T>
static void register_simd_vector_ge() {
  register_simd_vector_cmp<T, OP_GREATER_EQUAL>();
}

template <typename T>
static void register_simd_vector_lt() {
  register_simd_vector_cmp<T, OP_LESS>();
}

template <typename T>
static void register_simd_vector_le() {
  register_simd_vector_cmp<T, OP_LESS_EQUAL>();
}

template <typename T>
static void register_simd_vector_eq() {
  register_simd_vector_cmp<T, OP_EQUAL>();
}

template <typename T>
static void register_simd_vector_neq() {
  register_simd_vector_cmp<T, OP_NOT_EQUAL>();
}

template <typename T>
static void register_simd_vector_and() {
  register_simd_vector_cmp<T, OP_LOGIC_AND>();
}

template <typename T>
static void register_simd_vector_or() {
  register_simd_vector_cmp<T, OP_LOGIC_OR>();
}

template <typename T>
static void register_simd_vector_dot() {
  std::string simd_vector_func_name = fmt::format("simd_vector_dot_{}", get_dtype<T>().GetTypeString());
  T (*simd_f0)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_vector_dot<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f0);
  register_builtin_math_func("dot");
}

template <typename T>
static void register_simd_vector_iota() {
  std::string simd_vector_func_name = fmt::format("simd_vector_iota_{}", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f0)(T, uint32_t, uint32_t) = simd::simd_vector_iota<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f0);
  register_builtin_math_func("iota");
}

void init_builtin_simd_vector_funcs() {
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_add, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_sub, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_mul, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_div, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_mod, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t,
                             uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_gt, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_ge, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_lt, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_le, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_eq, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_neq, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_and, Bit)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_or, Bit)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_ternary, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_dot, float, double)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_iota, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
}
}  // namespace rapidudf