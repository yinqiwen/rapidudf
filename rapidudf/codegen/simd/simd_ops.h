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
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/optype.h"
#include "rapidudf/types/simd.h"

namespace rapidudf {
namespace simd {
enum ReuseFlag {
  REUSE_NONE = 0,
  REUSE_LEFT = 1,
  REUSE_RIGHT = 2,
  REUSE_FIRST = 1,
  REUSE_SECOND = 2,
  REUSE_THIRD = 3,
};
using VectorDataWithDType = std::pair<VectorData, DType>;
template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_op(Vector<T> left, Vector<T> right);
template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_vector_scalar_op(Vector<T> left, T right);
template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_scalar_vector_op(T left, Vector<T> right);

template <typename T, OpToken op>
Vector<T> simd_vector_unary_op(Vector<T> left);

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_op(Vector<R> a, Vector<T> b, Vector<T> c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_vector_scalar_op(Vector<R> a, Vector<T> b, T c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_vector_op(Vector<R> a, T b, Vector<T> c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_scalar_op(Vector<R> a, T b, T c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_vector_op(R a, Vector<T> b, Vector<T> c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_scalar_vector_op(R a, T b, Vector<T> c);
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_scalar_op(R a, Vector<T> b, T c);

template <typename T>
T simd_vector_dot(Vector<T> left, Vector<T> right);

template <typename T>
Vector<T> simd_vector_iota(T start, uint32_t n);

}  // namespace simd
}  // namespace rapidudf