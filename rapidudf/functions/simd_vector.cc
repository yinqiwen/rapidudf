/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "rapidudf/context/context.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/simd_vector.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace functions {

static void throw_vector_expression_ex(int line, StringView src_line, StringView msg) {
  throw VectorExpressionException(line, src_line, msg);
}

template <typename T>
static simd::Vector<T> new_simd_vector(Context& ctx, uint32_t n) {
  simd::VectorData vdata = ctx.NewSimdVector<T>(32, n);
  return simd::Vector<T>(vdata);
}
template <typename T>
static void register_new_simd_vector() {
  std::string func_name = GetFunctionName(kBuiltinNewSimdVector, get_dtype<T>());
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), new_simd_vector<T>);
}

template <typename T, OpToken op>
static void register_unary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(op, dtype.ToSimdVector());
  void (*simd_f)(const T*, T*) = simd_vector_unary_op<T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f);
}
template <typename T, OpToken op>
static void register_binary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(op, dtype.ToSimdVector());
  void (*simd_f)(const T*, const T*, T*) = simd_vector_binary_op<T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f);
}
template <typename T, OpToken op>
static void register_ternary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(op, dtype.ToSimdVector());
  void (*simd_f)(const T*, const T*, const T*, T*) = simd_vector_ternary_op<T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f);
}

template <typename T>
static void register_simd_vector_dot() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_DOT, dtype.ToSimdVector());
  T (*simd_f0)(simd::Vector<T>, simd::Vector<T>) = simd_vector_dot<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
  // register_builtin_function("dot");
}

template <typename T>
static void register_simd_vector_iota() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_IOTA, dtype);
  simd::Vector<T> (*simd_f0)(Context&, T, uint32_t) = simd_vector_iota<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
}

template <typename T>
static void register_simd_vector_sum() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SUM, dtype.ToSimdVector());
  T (*simd_f0)(simd::Vector<T>) = simd_vector_sum<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);

  func_name = GetFunctionName(OP_AVG, dtype.ToSimdVector());
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_vector_avg<T>);
}
template <typename T>
static void register_simd_vector_filter() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_FILTER, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f0)(Context& ctx, simd::Vector<T>, simd::Vector<Bit>) = simd_vector_filter;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
}

template <typename T>
static void register_simd_vector_gather() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_GATHER, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f0)(Context& ctx, simd::Vector<T>, simd::Vector<int32_t>) = simd_vector_gather;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
}

template <typename T>
static void register_simd_vector_sort() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SORT, dtype.ToSimdVector());
  void (*simd_f0)(Context&, simd::Vector<T>, bool) = simd_vector_sort<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);

  func_name = GetFunctionName(OP_SELECT, dtype.ToSimdVector());
  void (*simd_f1)(Context&, simd::Vector<T>, size_t, bool) = simd_vector_select<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f1);

  func_name = GetFunctionName(OP_TOPK, dtype.ToSimdVector());
  void (*simd_f2)(Context&, simd::Vector<T>, size_t, bool) = simd_vector_topk<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f2);

  func_name = GetFunctionName(OP_ARG_SORT, dtype.ToSimdVector());
  simd::Vector<size_t> (*simd_f3)(Context&, simd::Vector<T>, bool) = simd_vector_argsort<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f3);

  func_name = GetFunctionName(OP_ARG_SELECT, dtype.ToSimdVector());
  simd::Vector<size_t> (*simd_f4)(Context&, simd::Vector<T>, size_t, bool) = simd_vector_argselect<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f4);
}

template <typename K, typename V>
static void register_simd_vector_key_valye_sort() {
  DType key_dtype = get_dtype<K>();
  DType value_dtype = get_dtype<V>();
  std::string func_name = GetFunctionName(OP_SORT_KV, key_dtype.ToSimdVector(), value_dtype.ToSimdVector());
  void (*simd_f0)(Context&, simd::Vector<K>, simd::Vector<V>, bool) = simd_vector_sort_key_value<K, V>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);

  func_name = GetFunctionName(OP_SELECT_KV, key_dtype.ToSimdVector(), value_dtype.ToSimdVector());
  void (*simd_f1)(Context&, simd::Vector<K>, simd::Vector<V>, size_t, bool) = simd_vector_select_key_value<K, V>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f1);

  func_name = GetFunctionName(OP_TOPK_KV, key_dtype.ToSimdVector(), value_dtype.ToSimdVector());
  void (*simd_f2)(Context&, simd::Vector<K>, simd::Vector<V>, size_t, bool) = simd_vector_topk_key_value<K, V>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f2);
}

#define REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_SIMD_VECTOR_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define RUDF_STL_REFLECT_HELPER_INIT(r, STL_HELPER, i, TYPE) STL_HELPER<TYPE>::Init();
#define RUDF_STL_REFLECT_HELPER(STL_HELPER, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(RUDF_STL_REFLECT_HELPER_INIT, STL_HELPER, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE(r, op, i, type) register_unary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_UNARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define REGISTER_SIMD_VECTOR_BINARY_FUNC_WITH_TYPE(r, op, i, type) register_binary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_BINARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_BINARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE(r, op, i, type) register_unary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_UNARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define REGISTER_SIMD_VECTOR_TERNARY_FUNC_WITH_TYPE(r, op, i, type) register_ternary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_TERNARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_TERNARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define KEY_VALUE_SORT_DTYPES (uint32_t)(int32_t)(uint64_t)(int64_t)(float)(double)
#define RUDF_SIMD_VECTOR_SORT_KV_REGISTER(r, kv) \
  register_simd_vector_key_valye_sort<BOOST_PP_SEQ_ELEM(0, kv), BOOST_PP_SEQ_ELEM(1, kv)>();

void init_builtin_simd_vector_funcs() {
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinThrowVectorExprEx, throw_vector_expression_ex);
  REGISTER_SIMD_VECTOR_FUNCS(register_new_simd_vector, float, double, long double, int64_t, int32_t, int16_t, int8_t,
                             uint64_t, uint32_t, uint16_t, uint8_t, Bit, StringView)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_dot, float, double)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_iota, float, double, int64_t, int32_t, uint64_t, uint32_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_sum, float, double, int64_t, int32_t, uint64_t, uint32_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_filter, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t, StringView, Bit)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_gather, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t, StringView, Bit)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_sort, float, double, int64_t, int32_t, int16_t, uint64_t, uint32_t,
                             uint16_t);
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(RUDF_SIMD_VECTOR_SORT_KV_REGISTER, (KEY_VALUE_SORT_DTYPES)(KEY_VALUE_SORT_DTYPES))

  RUDF_STL_REFLECT_HELPER(reflect::SimdVectorHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                          int64_t, float, double, Bit, StringView)

  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_CEIL, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ROUND, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_RINT, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_TRUNC, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_EXP, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_EXPM1, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_EXP2, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_FLOOR, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_SQRT, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_LOG, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_LOG2, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_LOG10, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_LOG1P, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_SIN, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_COS, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_TAN, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ASIN, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ACOS, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ATAN, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_SINH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_COSH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_TANH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ASINH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ACOSH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ATANH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ERF, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ERFC, float, double)

  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_HYPOT, float, double)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_ATAN2, float, double)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_POW, float, double)
  // REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MULTIPLY, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
  // uint32_t,
  //                                   uint16_t, uint8_t)
  // REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_DIVIDE, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
  //                                   uint16_t, uint8_t)

  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_CLAMP, float, double, uint64_t, int64_t, uint32_t, int32_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_FMA, float, double, uint64_t, int64_t, uint32_t, int32_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_FMS, float, double, uint64_t, int64_t, uint32_t, int32_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_FNMA, float, double, uint64_t, int64_t, uint32_t, int32_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_FNMS, float, double, uint64_t, int64_t, uint32_t, int32_t)
}
}  // namespace functions
}  // namespace rapidudf