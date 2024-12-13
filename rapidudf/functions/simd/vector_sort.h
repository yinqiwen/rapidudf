/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
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
#include "rapidudf/types/vector.h"
namespace rapidudf {
namespace functions {
template <typename T>
void simd_vector_sort(Context& ctx, Vector<T> data, bool descending);
template <typename T>
void simd_vector_select(Context& ctx, Vector<T> data, size_t k, bool descending);
template <typename T>
void simd_vector_topk(Context& ctx, Vector<T> data, size_t k, bool descending);
template <typename T>
Vector<size_t> simd_vector_argsort(Context& ctx, Vector<T> data, bool descending);
template <typename T>
Vector<size_t> simd_vector_argselect(Context& ctx, Vector<T> data, size_t k, bool descending);

template <typename K, typename V>
void simd_vector_sort_key_value(Context& ctx, Vector<K> key, Vector<V> value, bool descending);
template <typename K, typename V>
void simd_vector_select_key_value(Context& ctx, Vector<K> key, Vector<V> value, size_t k, bool descending);
template <typename K, typename V>
void simd_vector_topk_key_value(Context& ctx, Vector<K> key, Vector<V> value, size_t k, bool descending);
}  // namespace functions
}  // namespace rapidudf