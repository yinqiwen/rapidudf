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
#include "rapidudf/context/context.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/scalar.h"

namespace rapidudf {
namespace simd {

template <OpToken op>
Column* simd_column_unary_op(Column* left);

template <OpToken op>
Column* simd_column_binary_op(Column* left, Column* right);
template <OpToken op>
Column* simd_column_binary_column_scalar_op(Column* left, Scalar* right);
template <OpToken op>
Column* simd_column_binary_scalar_column_op(Scalar* left, Column* right);

template <OpToken op>
Column* simd_column_ternary_op(Column* a, Column* b, Column* c);
template <OpToken op>
Column* simd_column_ternary_column_column_scalar_op(Column* a, Column* b, Scalar* c);
template <OpToken op>
Column* simd_column_ternary_column_scalar_column_op(Column* a, Scalar* b, Column* c);
template <OpToken op>
Column* simd_column_ternary_column_scalar_scalar_op(Column* a, Scalar* b, Scalar* c);
template <OpToken op>
Column* simd_column_ternary_scalar_column_column_op(Scalar* a, Column* b, Column* c);
template <OpToken op>
Column* simd_column_ternary_scalar_scalar_column_op(Scalar* a, Scalar* b, Column* c);
template <OpToken op>
Column* simd_column_ternary_scalar_column_scalar_op(Scalar* a, Column* b, Scalar* c);

Scalar* simd_column_sum(Column* a);
Scalar* simd_column_dot(Column* a, Column* b);
Column* simd_column_clone(Column* data);

Column* simd_column_filter(Column* data, Column* bits);
Column* simd_column_gather(Column* data, Column* indices);

void column_sort(Column* data, bool descending);
void column_select(Column* data, size_t k, bool descending);
void column_topk(Column* data, size_t k, bool descending);
Column* column_argsort(Column* data, bool descending);
Column* column_argselect(Column* data, size_t k, bool descending);
void column_sort_key_value(Column* key, Column* value, bool descending);
void column_select_key_value(Column* key, Column* value, size_t k, bool descending);
void column_topk_key_value(Column* key, Column* value, size_t k, bool descending);

}  // namespace simd
}  // namespace rapidudf