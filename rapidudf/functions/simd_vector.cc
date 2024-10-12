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

#include "rapidudf/context/context.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

#include "rapidudf/reflect/simd_vector.h"

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

#define REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_SIMD_VECTOR_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SIMD_VECTOR_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define RUDF_STL_REFLECT_HELPER_INIT(r, STL_HELPER, i, TYPE) STL_HELPER<TYPE>::Init();
#define RUDF_STL_REFLECT_HELPER(STL_HELPER, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(RUDF_STL_REFLECT_HELPER_INIT, STL_HELPER, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

void init_builtin_simd_vector_funcs() {
  RUDF_FUNC_REGISTER_WITH_NAME(kBuiltinThrowVectorExprEx, throw_vector_expression_ex);
  REGISTER_SIMD_VECTOR_FUNCS(register_new_simd_vector, float, double, long double, int64_t, int32_t, int16_t, int8_t,
                             uint64_t, uint32_t, uint16_t, uint8_t, Bit, StringView)

  RUDF_STL_REFLECT_HELPER(reflect::SimdVectorHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                          int64_t, float, double, Bit, StringView)
}
}  // namespace functions
}  // namespace rapidudf