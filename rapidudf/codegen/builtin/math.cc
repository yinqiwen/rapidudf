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
#include <fmt/core.h>
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cmath>
#include <complex>
#include <unordered_set>
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/codegen/simd/simd_ops.h"
#include "rapidudf/types/simd.h"

#define REGISTER_MATH_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_MATH_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_MATH_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
namespace rapidudf {
static std::unordered_set<std::string_view>& get_builtin_math_funcs() {
  static std::unordered_set<std::string_view> funcs;
  return funcs;
}
template <typename T>
static void register_abs() {
  T (*abs_f)(T) = &std::abs;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "abs_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("abs");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ABS>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_pow() {
  T (*abs_f)(T, T) = &std::pow;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "pow_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("pow");
}

template <typename T>
static void register_ceil() {
  T (*abs_f)(T) = &std::ceil;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "ceil_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("ceil");
}

template <typename T>
static void register_erf() {
  T (*abs_f)(T) = &std::erf;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "erf_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("erf");
}

template <typename T>
static void register_erfc() {
  T (*abs_f)(T) = &std::erfc;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "erfc_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("erfc");
}

template <typename T>
static void register_exp() {
  T (*abs_f)(T) = &std::exp;
  std::string func_name = "exp_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("exp");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_EXP>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_expm1() {
  T (*abs_f)(T) = &std::expm1;
  std::string func_name = "expm1_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("expm1");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_EXPM1>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_exp2() {
  T (*abs_f)(T) = &std::exp2;
  std::string func_name = "exp2_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("exp2");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_EXP>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_floor() {
  T (*abs_f)(T) = &std::expm1;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "floor_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("floor");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_FLOOR>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_sqrt() {
  T (*abs_f)(T) = &std::sqrt;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "sqrt_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("sqrt");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_SQRT>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T>
static void register_log() {
  T (*abs_f)(T) = &std::log;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "log_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_LOG>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_log10() {
  T (*abs_f)(T) = &std::log10;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "log10_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log10");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_LOG10>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_log1p() {
  T (*abs_f)(T) = &std::log1p;
  std::string func_name = "log1p_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log1p");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_LOG1P>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_log2() {
  T (*f)(T) = &std::log2;
  std::string func_name = "log2_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("log2");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_LOG2>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_hypot() {
  T (*f)(T, T) = &std::hypot;
  std::string func_name = "hypot_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("hypot");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, OP_HYPOT>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_sin() {
  T (*f)(T) = &std::sin;
  std::string func_name = "sin_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("sin");
  get_builtin_math_funcs().insert("atanh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_SIN>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_cos() {
  T (*f)(T) = &std::cos;
  std::string func_name = "cos_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_COS>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
  get_builtin_math_funcs().insert("cos");
}
template <typename T>
static void register_tan() {
  T (*f)(T) = &std::tan;
  std::string func_name = "tan_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("tan");
}
template <typename T>
static void register_asin() {
  T (*f)(T) = &std::asin;
  std::string func_name = "asin_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("asin");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ASIN>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_acos() {
  T (*f)(T) = &std::acos;
  std::string func_name = "acos_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("acos");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ACOS>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_atan() {
  T (*f)(T) = &std::atan;
  std::string func_name = "atan_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("atan");
}
template <typename T>
static void register_atan2() {
  T (*f)(T, T) = &std::atan2;
  std::string func_name = "atan2_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("atan2");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, OP_ATAN2>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_sinh() {
  T (*f)(T) = &std::sinh;
  std::string func_name = "sinh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("sinh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_SINH>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_cosh() {
  T (*f)(T) = &std::cosh;
  std::string func_name = "cosh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("cosh");
}
template <typename T>
static void register_tanh() {
  T (*f)(T) = &std::tanh;
  std::string func_name = "tanh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("tanh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_TANH>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_asinh() {
  T (*f)(T) = &std::asinh;
  std::string func_name = "asinh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("asinh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ASINH>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_acosh() {
  T (*f)(T) = &std::cosh;
  std::string func_name = "acosh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("acosh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ACOSH>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static void register_atanh() {
  T (*f)(T) = &std::tanh;
  std::string func_name = "atanh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("atanh");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_unary_op<T, OP_ATANH>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static T scalar_max(T left, T right) {
  return std::max(left, right);
}
template <typename T>
static void register_max() {
  T (*f)(T, T) = &scalar_max<T>;
  std::string func_name = "max_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("max");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, OP_MAX>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}
template <typename T>
static T scalar_min(T left, T right) {
  return std::min(left, right);
}
template <typename T>
static void register_min() {
  T (*f)(T, T) = &scalar_min<T>;
  std::string func_name = "min_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("min");
  std::string simd_vector_func_name = "simd_vector_" + func_name;
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, OP_MIN>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
}

template <typename T, OpToken op>
static void register_simd_vector_arithmetic() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_binary_op<T, T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
  simd_vector_func_name = fmt::format("simd_vector_scalar_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_ff)(simd::Vector<T>, T, bool, uint32_t) = simd::simd_binary_scalar_op<T, T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_ff);
}

template <typename T, OpToken op>
static void register_simd_vector_cmp() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<simd::Bit> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_binary_op<T, simd::Bit, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f);
  simd_vector_func_name = fmt::format("simd_vector_scalar_{}_{}", kOpTokenStrs[op], get_dtype<T>().GetTypeString());
  simd::Vector<simd::Bit> (*simd_ff)(simd::Vector<T>, T, bool, uint32_t) =
      simd::simd_binary_scalar_op<T, simd::Bit, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_ff);
}

template <typename T>
static void register_simd_vector_ternary() {
  std::string simd_vector_func_name =
      fmt::format("simd_vector_ternary_{}_scalar_scalar", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f0)(simd::Vector<simd::Bit>, T, T, uint32_t) = simd::simd_ternary_op_scalar_scalar<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f0);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_vector_vector", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f1)(simd::Vector<simd::Bit>, simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_ternary_op_vector_vector<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f1);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_vector_scalar", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f2)(simd::Vector<simd::Bit>, simd::Vector<T>, T, uint32_t) =
      simd::simd_ternary_op_vector_scalar<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f2);
  simd_vector_func_name = fmt::format("simd_vector_ternary_{}_scalar_vector", get_dtype<T>().GetTypeString());
  simd::Vector<T> (*simd_f3)(simd::Vector<simd::Bit>, T, simd::Vector<T>, uint32_t) =
      simd::simd_ternary_op_scalar_vector<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(simd_vector_func_name.c_str(), simd_f3);
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

bool is_builtin_math_func(std::string_view name) { return get_builtin_math_funcs().count(name) == 1; }

void init_builtin_math() {
  REGISTER_MATH_FUNCS(register_abs, float, double, int64_t, int32_t)
  REGISTER_MATH_FUNCS(register_max, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t,
                      uint8_t)
  REGISTER_MATH_FUNCS(register_pow, float, double)
  REGISTER_MATH_FUNCS(register_ceil, float, double)
  REGISTER_MATH_FUNCS(register_erf, float, double)
  REGISTER_MATH_FUNCS(register_erfc, float, double)
  REGISTER_MATH_FUNCS(register_exp, float, double)
  REGISTER_MATH_FUNCS(register_expm1, float, double)
  REGISTER_MATH_FUNCS(register_exp2, float, double)
  REGISTER_MATH_FUNCS(register_floor, float, double)
  REGISTER_MATH_FUNCS(register_sqrt, float, double)
  REGISTER_MATH_FUNCS(register_log, float, double)
  REGISTER_MATH_FUNCS(register_log10, float, double)
  REGISTER_MATH_FUNCS(register_log1p, float, double)
  REGISTER_MATH_FUNCS(register_log2, float, double)
  REGISTER_MATH_FUNCS(register_hypot, float, double)
  REGISTER_MATH_FUNCS(register_sin, float, double)
  REGISTER_MATH_FUNCS(register_cos, float, double)
  REGISTER_MATH_FUNCS(register_tan, float, double)
  REGISTER_MATH_FUNCS(register_asin, float, double)
  REGISTER_MATH_FUNCS(register_acos, float, double)
  REGISTER_MATH_FUNCS(register_atan, float, double)
  REGISTER_MATH_FUNCS(register_atan2, float, double)
  REGISTER_MATH_FUNCS(register_sinh, float, double)
  REGISTER_MATH_FUNCS(register_cosh, float, double)
  REGISTER_MATH_FUNCS(register_tanh, float, double)
  REGISTER_MATH_FUNCS(register_asinh, float, double)
  REGISTER_MATH_FUNCS(register_acosh, float, double)
  REGISTER_MATH_FUNCS(register_atanh, float, double)
  REGISTER_MATH_FUNCS(register_simd_vector_add, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_sub, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_mul, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_div, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_mod, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t,
                      uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_gt, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_ge, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_lt, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_le, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_eq, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_neq, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                      uint16_t, uint8_t)
  REGISTER_MATH_FUNCS(register_simd_vector_and, simd::Bit)
  REGISTER_MATH_FUNCS(register_simd_vector_or, simd::Bit)
  REGISTER_MATH_FUNCS(register_simd_vector_ternary, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                      uint32_t, uint16_t, uint8_t)
}
}  // namespace rapidudf