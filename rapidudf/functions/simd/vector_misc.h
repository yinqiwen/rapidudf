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
#include "rapidudf/types/simd/vector.h"
namespace rapidudf {
namespace functions {
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

template <typename T, OpToken op = OP_EQUAL>
int simd_vector_find(simd::Vector<T> data, T v);

}  // namespace functions
}  // namespace rapidudf