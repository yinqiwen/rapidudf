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
#include "rapidudf/builtin/simd_vector/ops.h"

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include <type_traits>
#include <variant>

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/simd_vector.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"
namespace rapidudf {
namespace simd {

template <OpToken op>
Column* simd_column_unary_op(Column* left) {
  auto& ctx = left->GetContext();
  return std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, TablePtr>) {
          THROW_LOGIC_ERR(fmt::format("Can NOT do {} with simd_table ptr", op));
        } else {
          using value_type = typename T::value_type;
          if constexpr (is_valid_operand<value_type>(op)) {
            Vector<value_type> ret = simd_vector_unary_op<value_type, op>(ctx, arg);
            return reflect::SimdVectorHelper<value_type>::to_column(ret, ctx);
          } else {
            THROW_LOGIC_ERR(fmt::format("Can NOT do {} with dtype:{}", op, get_dtype<T>()));
          }
        }
        return (Column*)0;
      },
      left->GetInternal());
}

#define DEFINE_SIMD_UNARY_OP_TEMPLATE(r, _, ii, op) template Column* simd_column_unary_op<op>(Column * left);
#define DEFINE_SIMD_UNARY_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_UNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_UNARY_OP(OP_NOT, OP_NEGATIVE, OP_SIN, OP_COS, OP_TAN, OP_SINH, OP_COSH, OP_TANH, OP_ASIN, OP_ACOS, OP_ATAN);
DEFINE_SIMD_UNARY_OP(OP_ASINH, OP_ACOSH, OP_ATANH, OP_EXP, OP_EXP2, OP_EXPM1, OP_LOG, OP_LOG2, OP_LOG10, OP_LOG1P,
                     OP_SQRT, OP_FLOOR, OP_CEIL, OP_ERF, OP_ERFC, OP_ABS, OP_ROUND, OP_TRUNC);

}  // namespace simd
}  // namespace rapidudf