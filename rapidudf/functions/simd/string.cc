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

#include "rapidudf/functions/simd/string.h"
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/string.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/highway.h"
#include "rapidudf/types/bit.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int simd_string_find_char_impl(std::string_view s, char ch) {
  const uint8_t* input = reinterpret_cast<const uint8_t*>(s.data());
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  const hn::Vec<D> cmp = hn::Set(d, static_cast<uint8_t>(ch));
  size_t idx = 0;
  size_t len = s.size();
  if (len >= N) {
    for (; idx <= len - N; idx += N) {
      const hn::Vec<D> v = hn::LoadU(d, input + idx);
      auto mask = hn::Eq(v, cmp);
      auto found = hn::FindFirstTrue(d, mask);
      if (found >= 0) {
        return found + idx;
      }
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == len)) return -1;
  const size_t remaining = len - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v = hn::LoadN(d, input + idx, remaining);
  auto mask = hn::Eq(v, cmp);
  auto found = hn::FindFirstTrue(d, mask);
  if (found >= 0 && static_cast<size_t>(found) < remaining) {
    return found + idx;
  }
  return -1;
}

std::vector<uint32_t> simd_string_split_by_char_impl(std::string_view s, char ch) {
  std::vector<uint32_t> sep_positions;
  sep_positions.reserve(16);
  const uint8_t* input = reinterpret_cast<const uint8_t*>(s.data());
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  const hn::Vec<D> cmp = hn::Set(d, static_cast<uint8_t>(ch));
  size_t idx = 0;
  size_t len = s.size();

  if (len >= N) {
    for (; idx <= len - N; idx += N) {
      const hn::Vec<D> v = hn::LoadU(d, input + idx);
      auto mask = hn::Eq(v, cmp);
      auto found = hn::FindFirstTrue(d, mask);
      while (found >= 0) {
        sep_positions.emplace_back(idx + found);
        auto tmp = hn::SetOnlyFirst(mask);
        mask = hn::And(mask, hn::Not(tmp));
        found = hn::FindFirstTrue(d, mask);
      }
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == len)) return sep_positions;
  const size_t remaining = len - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v = hn::LoadN(d, input + idx, remaining);
  auto mask = hn::Eq(v, cmp);
  auto found = hn::FindFirstTrue(d, mask);
  while (found >= 0) {
    if (static_cast<size_t>(found) >= remaining) {
      break;
    }
    sep_positions.emplace_back(idx + found);
    auto tmp = hn::SetOnlyFirst(mask);
    mask = hn::And(mask, hn::Not(tmp));
    found = hn::FindFirstTrue(d, mask);
  }
  return sep_positions;
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {
int simd_string_find_char(std::string_view s, char ch) {
  HWY_EXPORT_T(Table, simd_string_find_char_impl);
  return HWY_DYNAMIC_DISPATCH_T(Table)(s, ch);
}

std::vector<std::string_view> simd_string_split_by_char(std::string_view s, char ch) {
  HWY_EXPORT_T(Table, simd_string_split_by_char_impl);
  std::vector<uint32_t> sep_positions = HWY_DYNAMIC_DISPATCH_T(Table)(s, ch);
  std::vector<std::string_view> ss;
  ss.reserve(sep_positions.size());
  size_t last_pos = 0;
  for (auto pos : sep_positions) {
    if (pos != last_pos) {
      auto part = s.substr(last_pos, pos - last_pos);
      if (!part.empty()) {
        ss.emplace_back(part);
      }
    }
    last_pos = pos + 1;
  }
  return ss;
}

}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE