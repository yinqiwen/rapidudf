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
#include <complex>
#include <unordered_set>
#include "rapidudf/codegen/builtin/builtin.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"

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
}

template <typename T>
static void register_expm1() {
  T (*abs_f)(T) = &std::expm1;
  std::string func_name = "expm1_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("expm1");
}

template <typename T>
static void register_exp2() {
  T (*abs_f)(T) = &std::exp2;
  std::string func_name = "exp2_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("exp2");
}

template <typename T>
static void register_floor() {
  T (*abs_f)(T) = &std::expm1;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "floor_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("floor");
}

template <typename T>
static void register_sqrt() {
  T (*abs_f)(T) = &std::sqrt;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "sqrt_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  printf("#####register_sqrt %d\n", get_builtin_math_funcs().size());
  get_builtin_math_funcs().insert("sqrt");
}

template <typename T>
static void register_log() {
  T (*abs_f)(T) = &std::log;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "log_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log");
}
template <typename T>
static void register_log10() {
  T (*abs_f)(T) = &std::log10;
  std::string name = get_dtype<T>().GetTypeString();
  std::string func_name = "log10_" + name;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log10");
}
template <typename T>
static void register_log1p() {
  T (*abs_f)(T) = &std::log1p;
  std::string func_name = "log1p_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), abs_f);
  get_builtin_math_funcs().insert("log1p");
}
template <typename T>
static void register_log2() {
  T (*f)(T) = &std::log2;
  std::string func_name = "log2_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("log2");
}
template <typename T>
static void register_hypot() {
  T (*f)(T, T) = &std::hypot;
  std::string func_name = "hypot_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("hypot");
}
template <typename T>
static void register_sin() {
  T (*f)(T) = &std::sin;
  std::string func_name = "sin_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("sin");
}
template <typename T>
static void register_cos() {
  T (*f)(T) = &std::cos;
  std::string func_name = "cos_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
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
}
template <typename T>
static void register_acos() {
  T (*f)(T) = &std::acos;
  std::string func_name = "acos_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("acos");
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
}
template <typename T>
static void register_sinh() {
  T (*f)(T) = &std::sinh;
  std::string func_name = "sinh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("sinh");
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
}
template <typename T>
static void register_asinh() {
  T (*f)(T) = &std::asinh;
  std::string func_name = "asinh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("asinh");
}
template <typename T>
static void register_acosh() {
  T (*f)(T) = &std::cosh;
  std::string func_name = "acosh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("acosh");
}
template <typename T>
static void register_atanh() {
  T (*f)(T) = &std::tanh;
  std::string func_name = "atanh_" + get_dtype<T>().GetTypeString();
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
  get_builtin_math_funcs().insert("atanh");
}

bool is_builtin_math_func(std::string_view name) { return get_builtin_math_funcs().count(name) == 1; }

void init_builtin_math() {
  printf("#####init_builtin_math enter:%d\n", get_builtin_math_funcs().size());
  REGISTER_MATH_FUNCS(register_abs, float, double, int64_t, int32_t)
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
  printf("#####init_builtin_math exit:%d\n", get_builtin_math_funcs().size());
}
}  // namespace rapidudf