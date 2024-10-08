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

#include "rapidudf/types/eval_value-inl.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
EvalValue to_eval_value(OpToken op) { return EvalValue(op); }

EvalValue to_eval_value(bool bv) {
  EvalValue ev;
  ev.dtype = get_dtype<bool>().Control();
  ev.scalar_bv = bv;
  return ev;
}

EvalValue to_eval_value(StringView sv) {
  EvalValue ev;
  ev.dtype = get_dtype<StringView>().Control();
  ev.scalar_sv = sv;
  return ev;
}

template <typename T>
EvalValue to_eval_value(simd::Vector<T> v) {
  auto vector = v.RawData();
  auto dtype = get_dtype<simd::Vector<T>>().Control();
  return EvalValue(dtype, vector);
}
#define DEFINE_TO_EVAL_VECTOR_TEMPLATE(r, op, ii, TYPE) template EvalValue to_eval_value<TYPE>(simd::Vector<TYPE> v);
#define DEFINE_TO_EVAL_VECTOR(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_TO_EVAL_VECTOR_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_TO_EVAL_VECTOR(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t,
                      StringView, Bit);
}  // namespace rapidudf