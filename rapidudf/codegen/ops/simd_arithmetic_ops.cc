/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/codegen/ops/simd_arithmetic_ops.cc"  // NOLINT

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

namespace rapidudf {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

using T = float;

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
HWY_ATTR void AddLoop(const T* HWY_RESTRICT add_left, const T* HWY_RESTRICT add_right, const size_t size,
                      T* HWY_RESTRICT result) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto left = hn::Load(d, add_left + i);
    const auto right = hn::Load(d, add_right + i);
    auto x = hn::Add(left, right);
    hn::Store(x, d, result + i);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace rapidudf

// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace rapidudf {

// This macro declares a static array used for dynamic dispatch.
HWY_EXPORT(AddLoop);

void CallAddLoop(const float* HWY_RESTRICT left, const float* HWY_RESTRICT right, const size_t size,
                 float* HWY_RESTRICT result) {
  // This must reside outside of HWY_NAMESPACE because it references (calls the
  // appropriate one from) the per-target implementations there.
  // For static dispatch, use HWY_STATIC_DISPATCH.
  return HWY_DYNAMIC_DISPATCH(AddLoop)(left, right, size, result);
}

}  // namespace rapidudf
#endif  // HWY_ONCE
