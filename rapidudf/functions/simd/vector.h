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
#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/functions/simd/vector_op.h"
#include "rapidudf/functions/simd/vector_sort.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/vector.h"
namespace rapidudf {
namespace functions {

template <typename T, OpToken op>
void simd_vector_unary_op(const T* input, T* output);
template <typename T, OpToken op>
void simd_vector_binary_op(const T* left, const T* right, T* output);
template <typename T, OpToken op>
void simd_vector_ternary_op(const T* a, const T* b, const T* c, T* output);

}  // namespace functions
}  // namespace rapidudf