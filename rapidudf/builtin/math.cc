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

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cmath>

#include <unordered_set>
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"

#define REGISTER_MATH_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_MATH_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_MATH_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
namespace rapidudf {

static std::unordered_set<std::string_view>& get_builtin_math_funcs() {
  static std::unordered_set<std::string_view> funcs;
  return funcs;
}

static std::unordered_map<std::string, OpToken>& get_builtin_func_op_mapping() {
  static std::unordered_map<std::string, OpToken> mapping;
  return mapping;
}

void register_builtin_function_op(const std::string& name, OpToken op) { get_builtin_func_op_mapping()[name] = op; }

bool register_builtin_function(std::string_view name) { return get_builtin_math_funcs().insert(name).second; }

template <typename T>
static void register_abs() {
  T (*abs_f)(T) = &std::abs;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ABS, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_pow() {
  T (*abs_f)(T, T) = &std::pow;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_POW, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_ceil() {
  T (*abs_f)(T) = &std::ceil;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_CEIL, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_erf() {
  T (*abs_f)(T) = &std::erf;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ERF, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_erfc() {
  T (*abs_f)(T) = &std::erfc;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ERFC, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_exp() {
  T (*abs_f)(T) = &std::exp;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_EXP, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_EXP);
}

template <typename T>
static void register_expm1() {
  T (*abs_f)(T) = &std::expm1;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_EXPM1, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_exp2() {
  T (*abs_f)(T) = &std::exp2;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_EXP2, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_EXP2);
}

template <typename T>
static void register_floor() {
  T (*abs_f)(T) = &std::expm1;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_FLOOR, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}

template <typename T>
static void register_sqrt() {
  T (*abs_f)(T) = &std::sqrt;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SQRT, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_SQRT);
}

template <typename T>
static void register_log() {
  T (*abs_f)(T) = &std::log;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_LOG, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_LOG);
}
template <typename T>
static void register_log10() {
  T (*abs_f)(T) = &std::log10;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_LOG10, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_LOG10);
}
template <typename T>
static void register_log1p() {
  T (*abs_f)(T) = &std::log1p;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_LOG1P, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
}
template <typename T>
static void register_log2() {
  T (*f)(T) = &std::log2;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_LOG2, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  register_builtin_function_op(func_name, OP_LOG2);
}
template <typename T>
static void register_hypot() {
  T (*f)(T, T) = &std::hypot;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_HYPOT, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_sin() {
  T (*f)(T) = &std::sin;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SIN, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  register_builtin_function_op(func_name, OP_SIN);
}
template <typename T>
static void register_cos() {
  T (*f)(T) = &std::cos;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_COS, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  register_builtin_function_op(func_name, OP_COS);
}
template <typename T>
static void register_tan() {
  T (*f)(T) = &std::tan;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_TAN, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_asin() {
  T (*f)(T) = &std::asin;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ASIN, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_acos() {
  T (*f)(T) = &std::acos;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ACOS, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_atan() {
  T (*f)(T) = &std::atan;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ATAN, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}

template <typename T>
static void register_atan2() {
  T (*f)(T, T) = &std::atan2;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ATAN2, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_sinh() {
  T (*f)(T) = &std::sinh;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SINH, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_cosh() {
  T (*f)(T) = &std::cosh;
  std::string func_name = "cosh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  register_builtin_function("cosh");
}
template <typename T>
static void register_tanh() {
  T (*f)(T) = &std::tanh;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_TANH, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_asinh() {
  T (*f)(T) = &std::asinh;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ASINH, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_acosh() {
  T (*f)(T) = &std::acosh;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ACOSH, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static void register_atanh() {
  T (*f)(T) = &std::atanh;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_ATANH, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static T scalar_max(T left, T right) {
  return std::max(left, right);
}
template <typename T>
static void register_max() {
  T (*f)(T, T) = &scalar_max<T>;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_MAX, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}
template <typename T>
static T scalar_min(T left, T right) {
  return std::min(left, right);
}
template <typename T>
static void register_min() {
  T (*f)(T, T) = &scalar_min<T>;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_MIN, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}

template <typename T>
static T scalar_clamp(T a, T b, T c) {
  return std::clamp(a, b, c);
}

template <typename T>
static void register_fma() {
  T (*abs_f)(T, T, T) = &std::fma;
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_FMA, dtype);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  register_builtin_function_op(func_name, OP_FMA);
}

template <typename T>
static void register_clamp() {
  T (*f)(T, T, T) = &scalar_clamp<T>;
  std::string func_name = "clamp_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}

bool is_builtin_function(std::string_view name) { return get_builtin_math_funcs().count(name) == 1; }

OpToken get_buitin_func_op(const std::string& name) {
  auto found = get_builtin_func_op_mapping().find(name);
  if (found != get_builtin_func_op_mapping().end()) {
    return found->second;
  }
  return OP_INVALID;
}

void init_builtin_math_funcs() {
  for (uint32_t op = OP_CONDITIONAL; op < OP_END; op++) {
    get_builtin_math_funcs().insert(kOpTokenStrs[op]);
  }

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
  REGISTER_MATH_FUNCS(register_clamp, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t,
                      uint8_t)
  REGISTER_MATH_FUNCS(register_fma, float, double)
}
}  // namespace rapidudf