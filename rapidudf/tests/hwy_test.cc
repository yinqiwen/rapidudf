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