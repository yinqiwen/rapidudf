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
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include <stdexcept>
#include "rapidudf/builtin/simd_vector/column_ops.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/struct.h"
#include "rapidudf/types/simd_vector_table.h"
namespace rapidudf {

static void register_misc_simd_column_ops() {
  std::string func_name = GetFunctionName(OP_SUM, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::simd_column_sum);

  func_name = GetFunctionName(OP_DOT, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::simd_column_dot);

  // func_name = GetFunctionName(OP_IOTA, DATA_SIMD_COLUMN);
  // RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::simd_column_iota);

  func_name = GetFunctionName(OP_CLONE, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::simd_column_clone);
}

static void register_sort_simd_column_ops() {
  std::string func_name = GetFunctionName(OP_SORT, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_sort);
  func_name = GetFunctionName(OP_SELECT, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_select);
  func_name = GetFunctionName(OP_TOPK, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_topk);
  func_name = GetFunctionName(OP_ARG_SORT, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_argsort);
  func_name = GetFunctionName(OP_ARG_SELECT, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_argselect);

  func_name = GetFunctionName(OP_TOPK_KV, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_topk_key_value);
  func_name = GetFunctionName(OP_SORT_KV, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_sort_key_value);
  func_name = GetFunctionName(OP_SELECT_KV, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd::column_select_key_value);
}

template <OpToken op>
static void register_unary_simd_column_op() {
  std::string func_name = GetFunctionName(op, DATA_SIMD_COLUMN);
  simd::Column* (*simd_column_f)(simd::Column*) = simd::simd_column_unary_op<op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_column_f);
}

template <OpToken op>
static void register_binary_simd_column_op() {
  std::string func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
  simd::Column* (*simd_column_f)(simd::Column*, simd::Column*) = simd::simd_column_binary_op<op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_column_f);

  func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SCALAR);
  simd::Column* (*simd_column_f0)(simd::Column*, Scalar*) = simd::simd_column_binary_column_scalar_op<op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_column_f0);

  func_name = GetFunctionName(op, DATA_SCALAR, DATA_SIMD_COLUMN);
  simd::Column* (*simd_column_f1)(Scalar*, simd::Column*) = simd::simd_column_binary_scalar_column_op<op>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), simd_column_f1);
}

template <OpToken op>
static void register_ternary_simd_column_op() {
  std::string simd_vector_func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
  simd::Column* (*simd_f0)(simd::Column*, simd::Column*, simd::Column*) = simd::simd_column_ternary_op<op>;
  RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f0);

  simd_vector_func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN, DATA_SCALAR);
  simd::Column* (*simd_f1)(simd::Column*, simd::Column*, Scalar*) =
      simd::simd_column_ternary_column_column_scalar_op<op>;
  RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f1);

  simd_vector_func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SCALAR, DATA_SIMD_COLUMN);
  simd::Column* (*simd_f2)(simd::Column*, Scalar*, simd::Column*) =
      simd::simd_column_ternary_column_scalar_column_op<op>;
  RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f2);

  simd_vector_func_name = GetFunctionName(op, DATA_SIMD_COLUMN, DATA_SCALAR, DATA_SCALAR);
  simd::Column* (*simd_f3)(simd::Column*, Scalar*, Scalar*) = simd::simd_column_ternary_column_scalar_scalar_op<op>;
  RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f3);
  if constexpr (op != OP_CONDITIONAL) {
    simd_vector_func_name = GetFunctionName(op, DATA_SCALAR, DATA_SIMD_COLUMN, DATA_SIMD_COLUMN);
    simd::Column* (*simd_f4)(Scalar*, simd::Column*, simd::Column*) =
        simd::simd_column_ternary_scalar_column_column_op<op>;
    RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f4);

    simd_vector_func_name = GetFunctionName(op, DATA_SCALAR, DATA_SIMD_COLUMN, DATA_SCALAR);
    simd::Column* (*simd_f5)(Scalar*, simd::Column*, Scalar*) = simd::simd_column_ternary_scalar_column_scalar_op<op>;
    RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f5);

    simd_vector_func_name = GetFunctionName(op, DATA_SCALAR, DATA_SCALAR, DATA_SIMD_COLUMN);
    simd::Column* (*simd_f6)(Scalar*, Scalar*, simd::Column*) = simd::simd_column_ternary_scalar_scalar_column_op<op>;
    RUDF_FUNC_REGISTER_WITH_HASH_AND_NAME(op, simd_vector_func_name.c_str(), simd_f6);
  }
}

#define REGISTER_SIMD_COLUMN_UNARY_FUNC_WITH_TYPE(r, _, i, op) register_unary_simd_column_op<op>();
#define REGISTER_SIMD_COLUMN_UNARY_FUNCS(...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_COLUMN_UNARY_FUNC_WITH_TYPE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_COLUMN_BINARY_FUNC_WITH_TYPE(r, _, i, op) register_binary_simd_column_op<op>();
#define REGISTER_SIMD_COLUMN_BINARY_FUNCS(...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_COLUMN_BINARY_FUNC_WITH_TYPE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define REGISTER_SIMD_COLUMN_TERNARY_FUNC_WITH_TYPE(r, _, i, op) register_ternary_simd_column_op<op>();
#define REGISTER_SIMD_COLUMN_TERNARY_FUNCS(...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_COLUMN_TERNARY_FUNC_WITH_TYPE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

void init_builtin_simd_column_funcs() {
  register_misc_simd_column_ops();
  register_sort_simd_column_ops();
  REGISTER_SIMD_COLUMN_UNARY_FUNCS(OP_ABS, OP_CEIL, OP_EXP, OP_EXPM1, OP_EXP2, OP_FLOOR, OP_SQRT, OP_LOG, OP_LOG2,
                                   OP_LOG10, OP_LOG1P, OP_ROUND, OP_TRUNC)
  REGISTER_SIMD_COLUMN_UNARY_FUNCS(OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN, OP_SINH, OP_COSH, OP_TANH,
                                   OP_ASINH, OP_ACOSH, OP_ATANH, OP_ERF, OP_ERFC, OP_NOT, OP_NEGATIVE)
  REGISTER_SIMD_COLUMN_BINARY_FUNCS(OP_PLUS, OP_MINUS, OP_MULTIPLY, OP_DIVIDE, OP_MOD, OP_PLUS_ASSIGN, OP_MINUS_ASSIGN,
                                    OP_MULTIPLY_ASSIGN, OP_DIVIDE_ASSIGN, OP_MOD_ASSIGN, OP_MAX, OP_MIN)
  REGISTER_SIMD_COLUMN_BINARY_FUNCS(OP_HYPOT, OP_ATAN2, OP_POW, OP_GREATER, OP_GREATER_EQUAL, OP_LESS, OP_LESS_EQUAL,
                                    OP_EQUAL, OP_NOT_EQUAL, OP_LOGIC_AND, OP_LOGIC_OR)
  REGISTER_SIMD_COLUMN_TERNARY_FUNCS(OP_CONDITIONAL, OP_CLAMP, OP_FMA, OP_FMS, OP_FNMA, OP_FNMS);
}

}  // namespace rapidudf