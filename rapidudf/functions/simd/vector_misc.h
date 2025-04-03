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
#include "rapidudf/functions/simd/vector_op.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/vector.h"
namespace rapidudf {
namespace functions {
template <typename T>
T simd_vector_dot_distance(Vector<T> left, Vector<T> right);
template <typename T>
T simd_vector_cosine_distance(Vector<T> left, Vector<T> right);
template <typename T>
T simd_vector_l2_distance(Vector<T> left, Vector<T> right);

template <typename T>
Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n);
template <typename T>
T simd_vector_sum(Vector<T> left);
template <typename T>
T simd_vector_avg(Vector<T> left);
template <typename T>
T simd_vector_reduce_max(Vector<T> left);
template <typename T>
T simd_vector_reduce_min(Vector<T> left);

template <typename T>
Vector<T> simd_vector_gather(Context& ctx, Vector<T> data, Vector<int32_t> indices);
template <typename T>
Vector<T> simd_vector_filter(Context& ctx, Vector<T> data, Vector<Bit> bits);

template <typename T, OpToken op = OP_EQUAL>
int simd_vector_find(Vector<T> data, T v);

/**
 * return matched data count, store matched mask into param 'mask'
 */
template <typename T, OpToken op = OP_EQUAL>
size_t simd_vector_match(const T* data, size_t len, T v, uint64_t& mask);

template <typename T>
void simd_vector_random(Context& ctx, uint64_t seed, T* output);

template <typename T>
T random(uint64_t seed);

}  // namespace functions
}  // namespace rapidudf