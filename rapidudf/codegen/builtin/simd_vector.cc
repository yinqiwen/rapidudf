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
#include "rapidudf/reflect/simd_vector.h"
#include "rapidudf/types/simd.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
#define REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();
#define REGISTER_SIMD_VECTOR_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE(r, op, i, type) register_unary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_UNARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_UNARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_VECTOR_BINARY_FUNC_WITH_TYPE(r, op, i, type) register_binary_simd_vector_op<type, type, op>();
#define REGISTER_SIMD_VECTOR_BINARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_BINARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNC_WITH_TYPE(r, op, i, type) register_binary_simd_vector_op<type, Bit, op>();
#define REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_VECTOR_TERNARY_FUNC_WITH_TYPE(r, op, i, type) register_ternary_simd_vector_op<type, op>();
#define REGISTER_SIMD_VECTOR_TERNARY_FUNCS(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_TERNARY_FUNC_WITH_TYPE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define RUDF_STL_REFLECT_HELPER_INIT(r, STL_HELPER, i, TYPE) STL_HELPER<TYPE>::Init();
#define RUDF_STL_REFLECT_HELPER(STL_HELPER, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(RUDF_STL_REFLECT_HELPER_INIT, STL_HELPER, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

template <typename T, OpToken op>
static void register_unary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(op, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f)(simd::Vector<T>, uint32_t) = simd::simd_vector_unary_op<T, op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f);
}

template <typename T, typename R, OpToken op>
static void register_binary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(op, dtype.ToSimdVector(), dtype.ToSimdVector());
  simd::Vector<R> (*simd_f)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_vector_binary_op<T, R, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, func_name.c_str(), simd_f);
  func_name = GetFunctionName(op, dtype.ToSimdVector(), dtype);
  simd::Vector<R> (*simd_f1)(simd::Vector<T>, T, uint32_t) = simd::simd_vector_binary_vector_scalar_op<T, R, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, func_name.c_str(), simd_f1);
  func_name = GetFunctionName(op, dtype, dtype.ToSimdVector());
  simd::Vector<R> (*simd_f2)(T, simd::Vector<T>, uint32_t) = simd::simd_vector_binary_scalar_vector_op<T, R, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(op, func_name.c_str(), simd_f2);
}

template <typename T, OpToken op>
static void register_ternary_simd_vector_op() {
  DType dtype = get_dtype<T>();
  std::string simd_vector_func_name =
      GetFunctionName(op, dtype.ToSimdVector(), dtype.ToSimdVector(), dtype.ToSimdVector());
  simd::Vector<T> (*simd_f0)(simd::Vector<T>, simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f0);
  simd_vector_func_name = GetFunctionName(op, dtype.ToSimdVector(), dtype.ToSimdVector(), dtype);
  simd::Vector<T> (*simd_f1)(simd::Vector<T>, simd::Vector<T>, T, uint32_t) =
      simd::simd_vector_ternary_vector_vector_scalar_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f1);
  simd_vector_func_name = GetFunctionName(op, dtype.ToSimdVector(), dtype, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f2)(simd::Vector<T>, T, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_vector_scalar_vector_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f2);
  simd_vector_func_name = GetFunctionName(op, dtype.ToSimdVector(), dtype, dtype);
  simd::Vector<T> (*simd_f3)(simd::Vector<T>, T, T, uint32_t) =
      simd::simd_vector_ternary_vector_scalar_scalar_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f3);

  simd_vector_func_name = GetFunctionName(op, dtype, dtype.ToSimdVector(), dtype.ToSimdVector());
  simd::Vector<T> (*simd_f4)(T, simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_scalar_vector_vector_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f4);
  simd_vector_func_name = GetFunctionName(op, dtype, dtype.ToSimdVector(), dtype);
  simd::Vector<T> (*simd_f5)(T, simd::Vector<T>, T, uint32_t) =
      simd::simd_vector_ternary_scalar_vector_scalar_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f5);
  simd_vector_func_name = GetFunctionName(op, dtype, dtype, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f6)(T, T, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_scalar_scalar_vector_op<T, T, op>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f6);
}

template <typename T>
static void register_ternary_conditional_simd_vector_op() {
  DType dtype = get_dtype<T>();
  DType bits_dtype = DType(DATA_BIT).ToSimdVector();
  std::string simd_vector_func_name =
      GetFunctionName(OP_CONDITIONAL, bits_dtype, dtype.ToSimdVector(), dtype.ToSimdVector());
  simd::Vector<T> (*simd_f0)(simd::Vector<Bit>, simd::Vector<T>, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_op<Bit, T, OP_CONDITIONAL>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f0);

  simd_vector_func_name = GetFunctionName(OP_CONDITIONAL, bits_dtype, dtype.ToSimdVector(), dtype);
  simd::Vector<T> (*simd_f1)(simd::Vector<Bit>, simd::Vector<T>, T, uint32_t) =
      simd::simd_vector_ternary_vector_vector_scalar_op<Bit, T, OP_CONDITIONAL>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f1);

  simd_vector_func_name = GetFunctionName(OP_CONDITIONAL, bits_dtype, dtype, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f2)(simd::Vector<Bit>, T, simd::Vector<T>, uint32_t) =
      simd::simd_vector_ternary_vector_scalar_vector_op<Bit, T, OP_CONDITIONAL>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f2);

  simd_vector_func_name = GetFunctionName(OP_CONDITIONAL, bits_dtype, dtype, dtype);
  simd::Vector<T> (*simd_f3)(simd::Vector<Bit>, T, T, uint32_t) =
      simd::simd_vector_ternary_vector_scalar_scalar_op<Bit, T, OP_CONDITIONAL>;
  RUDF_SAFE_FUNC_REGISTER_WITH_HASH_AND_NAME(0, simd_vector_func_name.c_str(), simd_f3);
}

template <typename T>
static void register_simd_vector_dot() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_DOT, dtype.ToSimdVector(), dtype.ToSimdVector());
  T (*simd_f0)(simd::Vector<T>, simd::Vector<T>, uint32_t) = simd::simd_vector_dot<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
  register_builtin_math_func("dot");
}

template <typename T>
static void register_simd_vector_iota() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_IOTA, dtype.ToSimdVector());
  simd::Vector<T> (*simd_f0)(T, uint32_t, uint32_t) = simd::simd_vector_iota<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_f0);
  register_builtin_math_func("iota");
}

void init_builtin_simd_vector_funcs() {
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ABS, float, double, int64_t, int32_t, int16_t, int8_t)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_CEIL, float, double)
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
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ASIN, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ACOS, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_SINH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_TANH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ASINH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ACOSH, float, double)
  REGISTER_SIMD_VECTOR_UNARY_FUNCS(OP_ATANH, float, double)

  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_PLUS, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MINUS, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MULTIPLY, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_DIVIDE, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MOD, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_GREATER_EQUAL, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                         uint32_t, uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_GREATER, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                         uint32_t, uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_LESS_EQUAL, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                         uint32_t, uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_LESS, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                         uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_NOT_EQUAL, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                         uint32_t, uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_BOOL_FUNCS(OP_EQUAL, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                         uint16_t, uint8_t, StringView)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_LOGIC_AND, Bit)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_LOGIC_OR, Bit)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_HYPOT, float, double)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_ATAN2, float, double)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MAX, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_BINARY_FUNCS(OP_MIN, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                    uint16_t, uint8_t)

  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_CLAMP, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                     uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_MULADD, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                     uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_MULSUB, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                     uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_MULADDSUB, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                                     uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_NEG_MULADD, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                     uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_TERNARY_FUNCS(OP_NEG_MULSUB, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                                     uint32_t, uint16_t, uint8_t)

  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_dot, float, double)
  REGISTER_SIMD_VECTOR_FUNCS(register_simd_vector_iota, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t,
                             uint32_t, uint16_t, uint8_t)
  REGISTER_SIMD_VECTOR_FUNCS(register_ternary_conditional_simd_vector_op, float, double, int64_t, int32_t, int16_t,
                             int8_t, uint64_t, uint32_t, uint16_t, uint8_t)

  RUDF_STL_REFLECT_HELPER(reflect::SimdVectorHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                          int64_t, float, double, Bit, StringView)
}
}  // namespace rapidudf