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

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include "hwy/highway.h"
#include "rapidudf/log/log.h"
namespace hn = hwy::HWY_NAMESPACE;

TEST(SIMD, mask) {
  using D = hn::ScalableTag<int32_t>;
  using M = hn::MFromD<D>;

  using D2 = hn::Half<D>;
  using M2 = hn::MFromD<D2>;

  using D3 = hn::ScalableTag<int64_t>;
  using M3 = hn::MFromD<D3>;
  D d;

  M2 m2;
  M3 m3;
  m2 = hn::DemoteMaskTo(D2{}, D3{}, m3);

  M m = hn::CombineMasks(d, m2, m2);

  RUDF_INFO("{} {}", hn::Lanes(D{}), hn::Lanes(D2{}));
}

TEST(SIMD, rebind) {
  using D = hn::ScalableTag<uint32_t>;
  hn::Rebind<uint8_t, D> d8;

  uint32_t t[32];
  uint8_t t1[32];
  for (size_t i = 0; i < 32; i++) {
    t[i] = i;
    t1[i] = 0;
  }

  const D d;
  constexpr auto lanes = hn::Lanes(d);

  size_t idx = 0;
  for (; (idx + lanes) <= 32; idx += lanes) {
    const hn::Vec<D> v = hn::LoadU(d, t + idx);
    auto period_u8 = hn::U8FromU32(v);
    hn::StoreU(period_u8, d8, t1 + idx);
    // auto period = hn::MaskedDiv(mask, elapsed, decay_val);
    // auto period_u8 = hn::U8FromU32(period);
  }

  RUDF_INFO("{} {}", hn::Lanes(D{}), hn::Lanes(d8));
  for (size_t i = 0; i < 32; i++) {
    RUDF_INFO("{} {}", t1[i], t[i]);
  }
}