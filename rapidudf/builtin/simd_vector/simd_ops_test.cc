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
#include "rapidudf/builtin/simd_vector/simd_ops.h"
#include <gtest/gtest.h>
#include <functional>
#include <vector>

#include "hwy/highway.h"
#include "rapidudf/log/log.h"
namespace hn = hwy::HWY_NAMESPACE;
using namespace rapidudf;
// TEST(SIMD, simple) {
//   const hn::ScalableTag<double> d;
//   auto v1 = hn::Set(d, 2);
//   auto v2 = hn::Zero(d);
//   using TTT = hn::TFromV<decltype(v1)>;
//   using MaskType = hn::Mask<decltype(d)>;
//   RUDF_INFO("mask size:{} lanes:{}", sizeof(MaskType), hn::Lanes(d));
//   auto mask = v1 != v2;
//   uint8_t bits[8];
//   size_t n = hn::StoreMaskBits(d, mask, bits);
//   RUDF_INFO("bits n:{} {}", n, bits[0]);

//   RUDF_INFO("double lanes:{}", hn::Lanes(d));
//   const hn::ScalableTag<float> d1;
//   RUDF_INFO("float lanes:{}", hn::Lanes(d1));
//   const hn::ScalableTag<short> d2;
//   RUDF_INFO("short lanes:{}", hn::Lanes(d2));
//   const hn::ScalableTag<uint8_t> d3;
//   RUDF_INFO("uint8_t lanes:{}", hn::Lanes(d3));

//   RUDF_INFO("{}", hwy::TargetName(HWY_TARGET));

//   const hn::ScalableTag<uint8_t> u8_d;
//   const hn::ScalableTag<uint32_t> u32_d;
//   const hn::ScalableTag<float> f32_d;
//   const hn::ScalableTag<uint64_t> u64_d;
//   const hn::ScalableTag<double> f64_d;
//   auto u8_v = hn::Set(u8_d, 2);
//   auto u32_v = hn::Set(u32_d, 2);
//   auto u64_v = hn::Set(u64_d, 2);
//   auto f32_v = hn::Set(f32_d, 2);
//   auto f64_v = hn::Set(f64_d, 2);
//   hn::ConvertTo(f32_d, u32_v);

//   using DF = hn::ScalableTag<float>;

//   const hn::RebindToSigned<DF> d32;
//   const hn::Rebind<uint8_t, DF> d8;
//   const uint8_t* HWY_RESTRICT values = nullptr;
//   hn::PromoteTo(d32, hn::Load(d8, values));
//   // hn::TruncateTo(u32_d, u64_v);
// }

// TEST(SIMD, simple_op) {
//   std::vector<float> a{1.1, 2.2, 3.3, 4.4, 5, 6, 7, 8, 91, 92};
//   std::vector<float> b{10.1, 2.2, 3.3, 40.4, 50, 60, 70, 80, 90, 93};
//   std::vector<float> c{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//   auto final0 = simd::simd_binary_op<float, simd::Bit, OP_LESS>(a, b, simd::REUSE_NONE);
//   // auto [vec0, dtype] = std::move(result0.value());
//   // simd::Vector<simd::Bit> final0(vec0);
//   auto final1 = simd::simd_binary_op<float, simd::Bit, OP_GREATER>(b, c, simd::REUSE_NONE);
//   // auto [vec1, dtype1] = std::move(result1.value());
//   // simd::Vector<simd::Bit> final1(vec1);
//   // RUDF_INFO("{}", dtype);
//   for (size_t i = 0; i < final0.Size(); i++) {
//     RUDF_INFO("final0:{}", final0[i].val);
//   }
//   // RUDF_INFO("{}", dtype1);
//   for (size_t i = 0; i < final1.Size(); i++) {
//     RUDF_INFO("final1:{}", final1[i].val);
//   }
//   auto final2 = simd::simd_binary_op<simd::Bit, simd::Bit, OP_LOGIC_OR>(final0, final1, simd::REUSE_NONE);
//   // auto [vec2, dtype2] = std::move(result2.value());
//   // RUDF_INFO("{}", dtype2);
//   // simd::Vector<simd::Bit> final2(vec2);
//   for (size_t i = 0; i < final2.Size(); i++) {
//     RUDF_INFO("final2:{}", final2[i].val);
//   }
// }

TEST(SIMD, simple_op1) {
  const hn::ScalableTag<double> d;
  using MaskType = hn::Mask<decltype(d)>;
  constexpr auto lanes = hn::Lanes(d);

  RUDF_INFO("lanes:{}", lanes);
}