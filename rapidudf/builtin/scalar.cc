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
#include "rapidudf/types/scalar.h"
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cmath>

#include <unordered_set>
#include "rapidudf/builtin/builtin.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/reflect/macros.h"

namespace rapidudf {

RUDF_STRUCT_MEMBER_METHODS(Scalar, to_f64, to_f32, to_u64, to_u32, to_u16, to_u8, to_i64, to_i32, to_i16, to_i8, to_bit,
                           to_string_view);

template <typename T>
static void register_to_scalar() {
  DType dtype = get_dtype<T>();
  std::string func_name = GetFunctionName(OP_SCALAR_CAST, dtype);
  Scalar* (*f)(Context&, T) = &to_scalar<T>;
  RUDF_FUNC_REGISTER_WITH_NAME(func_name.c_str(), f);
}

#define REGISTER_SCALAR_FUNC_WITH_TYPE(r, FUNC, i, type) FUNC<type>();

#define REGISTER_SCALAR_FUNCS(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(REGISTER_SCALAR_FUNC_WITH_TYPE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

void init_builtin_scalar_funcs() {
  REGISTER_SCALAR_FUNCS(register_to_scalar, float, double, int64_t, int32_t, int16_t, int8_t, uint64_t, uint32_t,
                        uint16_t, uint8_t, StringView)
}
}  // namespace rapidudf