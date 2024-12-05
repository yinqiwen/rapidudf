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
#include "rapidudf/functions/simd/vector_op.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/vector/vector.h"

namespace rapidudf {
namespace functions {
void simd_vector_bits_not(simd::Vector<Bit> src, simd::Vector<Bit> dst);
void simd_vector_bits_and(simd::Vector<Bit> left, simd::Vector<Bit> right, simd::Vector<Bit> dst);
void simd_vector_bits_or(simd::Vector<Bit> left, simd::Vector<Bit> right, simd::Vector<Bit> dst);
void simd_vector_bits_xor(simd::Vector<Bit> left, simd::Vector<Bit> right, simd::Vector<Bit> dst);
}  // namespace functions
}  // namespace rapidudf