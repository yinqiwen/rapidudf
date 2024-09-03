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

#pragma once
#include "absl/status/statusor.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/types/simd.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace simd {
enum ReuseFlag {
  REUSE_NONE = 0,
  REUSE_LEFT = 1,
  REUSE_RIGHT = 2,
};
using VectorDataWithDType = std::pair<VectorData, DType>;
template <typename T, typename R, OpToken op>
Vector<R> simd_binary_op(Vector<T> left, Vector<T> right, uint32_t reuse);
template <typename T, typename R, OpToken op>
Vector<R> simd_binary_scalar_op(Vector<T> left, T right, bool reverse, uint32_t reuse);

template <typename T, OpToken op>
Vector<T> simd_unary_op(Vector<T> left, uint32_t reuse);

template <typename T>
Vector<T> simd_ternary_op_scalar_scalar(Vector<Bit> cond, T true_val, T false_val, uint32_t reuse);
template <typename T>
Vector<T> simd_ternary_op_vector_vector(Vector<Bit> cond, Vector<T> true_val, Vector<T> false_val, uint32_t reuse);
template <typename T>
Vector<T> simd_ternary_op_vector_scalar(Vector<Bit> cond, Vector<T> true_val, T false_val, uint32_t reuse);
template <typename T>
Vector<T> simd_ternary_op_scalar_vector(Vector<Bit> cond, T true_val, Vector<T> false_val, uint32_t reuse);

template <OpToken op>
Vector<Bit> simd_vector_string_cmp(Vector<StringView> left, Vector<StringView> right, uint32_t reuse);
template <OpToken op>
Vector<Bit> simd_vector_string_cmp_scalar(Vector<StringView> left, StringView right, bool reverse, uint32_t reuse);

template <typename T>
T simd_vector_dot(Vector<T> left, Vector<T> right, uint32_t reuse);

template <typename T>
Vector<T> simd_vector_iota(T start, uint32_t n, uint32_t reuse);

void init_builtin_simd_funcs();

}  // namespace simd
}  // namespace rapidudf