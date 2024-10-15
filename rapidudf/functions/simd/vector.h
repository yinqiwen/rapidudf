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
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd/vector.h"
namespace rapidudf {
namespace functions {
template <typename T, OpToken opt>
struct OperandType {
  using operand_t = T;
  static constexpr OpToken op = opt;
};
template <typename T, OpToken op>
void simd_vector_unary_op(const T* input, T* output);
template <typename T, OpToken op>
void simd_vector_binary_op(const T* left, const T* right, T* output);
template <typename T, OpToken op>
void simd_vector_ternary_op(const T* a, const T* b, const T* c, T* output);

template <typename T>
T simd_vector_dot(simd::Vector<T> left, simd::Vector<T> right);
template <typename T>
simd::Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n);
template <typename T>
T simd_vector_sum(simd::Vector<T> left);
template <typename T>
T simd_vector_avg(simd::Vector<T> left);
template <typename T>
simd::Vector<T> simd_vector_gather(Context& ctx, simd::Vector<T> data, simd::Vector<int32_t> indices);
template <typename T>
simd::Vector<T> simd_vector_filter(Context& ctx, simd::Vector<T> data, simd::Vector<Bit> bits);

template <typename T>
void simd_vector_sort(Context& ctx, simd::Vector<T> data, bool descending);
template <typename T>
void simd_vector_select(Context& ctx, simd::Vector<T> data, size_t k, bool descending);
template <typename T>
void simd_vector_topk(Context& ctx, simd::Vector<T> data, size_t k, bool descending);
template <typename T>
simd::Vector<size_t> simd_vector_argsort(Context& ctx, simd::Vector<T> data, bool descending);
template <typename T>
simd::Vector<size_t> simd_vector_argselect(Context& ctx, simd::Vector<T> data, size_t k, bool descending);

template <typename K, typename V>
void simd_vector_sort_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, bool descending);
template <typename K, typename V>
void simd_vector_select_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, size_t k, bool descending);
template <typename K, typename V>
void simd_vector_topk_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, size_t k, bool descending);

}  // namespace functions
}  // namespace rapidudf