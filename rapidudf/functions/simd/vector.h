/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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