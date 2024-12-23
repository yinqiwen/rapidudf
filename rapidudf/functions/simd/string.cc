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

#include "rapidudf/functions/simd/string.h"
#include <string.h>
#include <limits>
#include <vector>
#include "rapidudf/log/log.h"

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

HWY_INLINE int simd_string_find_char_impl(const char* s, size_t len, char ch) {
  const uint8_t* input = reinterpret_cast<const uint8_t*>(s);
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  const hn::Vec<D> cmp = hn::Set(d, static_cast<uint8_t>(ch));
  size_t idx = 0;

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

HWY_INLINE int simd_string_find_string_impl(const char* s, size_t len, const char* part, size_t part_len) {
  if (len < part_len) {
    return -1;
  }
  if (part_len == 1) {
    return simd_string_find_char_impl(s, len, part[0]);
  }
  const uint8_t* input = reinterpret_cast<const uint8_t*>(s);
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  size_t first_pos = 0;

  size_t last_pos = part_len - 1;
  const hn::Vec<D> cmp_first = hn::Set(d, static_cast<uint8_t>(part[first_pos]));
  const hn::Vec<D> cmp_last = hn::Set(d, static_cast<uint8_t>(part[last_pos]));
  size_t idx = 0;
  size_t min_simd_block_size = N + part_len - 1;

  if (len >= min_simd_block_size) {
    for (; idx <= len - min_simd_block_size; idx += N) {
      const hn::Vec<D> v0 = hn::LoadU(d, input + idx + first_pos);
      const hn::Vec<D> v1 = hn::LoadU(d, input + idx + last_pos);
      auto mask0 = hn::Eq(v0, cmp_first);
      auto mask1 = hn::Eq(v1, cmp_last);
      auto mask = hn::And(mask0, mask1);
      auto found = hn::FindFirstTrue(d, mask);
      while (found >= 0) {
        if (part_len == 2) {
          return found + idx;
        }
        if (memcmp(input + idx + first_pos + 1, part + 1, part_len - 2) == 0) {
          return found + idx;
        }
        auto tmp = hn::SetOnlyFirst(mask);
        mask = hn::And(mask, hn::Not(tmp));
        found = hn::FindFirstTrue(d, mask);
      }
    }
  }
  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == (len - part_len))) {
    return -1;
  }
  const size_t remaining = len - idx - part_len + 1;

  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v0 = hn::LoadN(d, input + idx, remaining);
  const hn::Vec<D> v1 = hn::LoadN(d, input + idx + last_pos, remaining);
  auto mask0 = hn::Eq(v0, cmp_first);
  auto mask1 = hn::Eq(v1, cmp_last);
  auto mask = hn::And(mask0, mask1);
  auto found = hn::FindFirstTrue(d, mask);
  while (found >= 0) {
    if (part_len == 2) {
      return found + idx;
    }
    if (memcmp(input + idx + found + 1, part + 1, part_len - 2) == 0) {
      return found + idx;
    }
    auto tmp = hn::SetOnlyFirst(mask);
    mask = hn::And(mask, hn::Not(tmp));
    found = hn::FindFirstTrue(d, mask);
  }
  return -1;
}

template <typename T>
struct ClearMaskHelper {
  using D = hn::ScalableTag<T>;
  using Mask = hn::Mask<D>;
  static HWY_INLINE Mask GetClearMask(size_t i) {
    constexpr D d;
    constexpr size_t N = hn::Lanes(d);
    static Mask masks[N];
    static bool inited = false;
    if (!inited) {
      T data[N];
      memset(data, 0, N);
      for (size_t pos_to_clear = 0; pos_to_clear < N; pos_to_clear++) {
        data[pos_to_clear] = std::numeric_limits<T>::max();
        auto clear_mask = hn::MaskFromVec(hn::LoadU(d, data));
        masks[pos_to_clear] = clear_mask;
        data[pos_to_clear] = 0;
      }
      inited = true;
    }
    return masks[i];
  }
};

HWY_INLINE std::vector<uint32_t> simd_string_split_by_char_impl(std::string_view s, char ch) {
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
        // auto tmp = ClearMaskHelper<uint8_t>::GetClearMask(found);
        mask = hn::AndNot(tmp, mask);
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
    // auto tmp = ClearMaskHelper<uint8_t>::GetClearMask(found);
    mask = hn::AndNot(tmp, mask);
    found = hn::FindFirstTrue(d, mask);
  }
  return sep_positions;
}

HWY_INLINE std::vector<uint32_t> simd_string_split_by_string_impl(std::string_view s, std::string_view sep) {
  if (sep.size() == 1) {
    return simd_string_split_by_char_impl(s, sep[0]);
  }
  std::vector<uint32_t> sep_positions;
  sep_positions.reserve(16);
  const char* data = s.data();
  size_t len = s.length();
  int idx = simd_string_find_string_impl(data, len, sep.data(), sep.size());
  uint32_t offset = 0;
  while (idx >= 0) {
    sep_positions.emplace_back(offset + idx);
    data += (idx + sep.size());
    offset += (idx + sep.size());
    len -= (idx + sep.size());
    idx = simd_string_find_string_impl(data, len, sep.data(), sep.size());
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
  return HWY_DYNAMIC_DISPATCH_T(Table)(s.data(), s.size(), ch);
}
int simd_string_find_string(std::string_view s, std::string_view part) {
  HWY_EXPORT_T(Table, simd_string_find_string_impl);
  return HWY_DYNAMIC_DISPATCH_T(Table)(s.data(), s.size(), part.data(), part.size());
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

std::vector<std::string_view> simd_string_split_by_string(std::string_view s, std::string_view sep) {
  HWY_EXPORT_T(Table, simd_string_split_by_string_impl);
  std::vector<uint32_t> sep_positions = HWY_DYNAMIC_DISPATCH_T(Table)(s, sep);

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
    last_pos = pos + sep.size();
  }
  return ss;
}

}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE