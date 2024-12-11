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
#include <string.h>
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

  if (len >= (N + part_len)) {
    for (; idx <= len - N - part_len; idx += N) {
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
  const size_t remaining = len - idx - part_len;

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

// HWY_INLINE std::vector<uint32_t> simd_string_split_by_char_impl(std::string_view s, char ch) {
//   std::vector<uint32_t> sep_positions;
//   sep_positions.reserve(16);
//   const char* data = s.data();
//   size_t len = s.length();
//   int idx = simd_string_find_char_impl(data, len, ch);
//   uint32_t offset = 0;
//   while (idx >= 0) {
//     sep_positions.emplace_back(offset + idx);
//     data += (idx + 1);
//     offset += (idx + 1);
//     len -= (idx + 1);
//     idx = simd_string_find_char_impl(data, len, ch);
//   }
//   return sep_positions;
// }

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

HWY_INLINE std::vector<std::string_view> simd_string_split_by_char_impl1(std::string_view s, char ch) {
  const uint8_t* input = reinterpret_cast<const uint8_t*>(s.data());
  using D = hn::ScalableTag<uint8_t>;
  constexpr D d;
  constexpr size_t N = hn::Lanes(d);
  const hn::Vec<D> cmp = hn::Set(d, static_cast<uint8_t>(ch));
  size_t idx = 0;
  size_t len = s.size();
  std::vector<uint8_t> indices(s.size());
  auto indice_vec = hn::Iota(d, 0);

  size_t indice_offset = 0;
  if (len >= N) {
    for (; idx <= len - N; idx += N) {
      const hn::Vec<D> v = hn::LoadU(d, input + idx);
      auto mask = hn::Eq(v, cmp);
      size_t n = hn::CompressStore(indice_vec, mask, d, indices.data() + indice_offset + 1);
      indices[indice_offset] = static_cast<uint8_t>(n);
      indice_offset += (n + 1);
    }
  }
  if (idx < len) {
    const size_t remaining = len - idx;
    HWY_DASSERT(0 != remaining && remaining < N);
    const hn::Vec<D> v = hn::LoadN(d, input + idx, remaining);
    auto mask = hn::Eq(v, cmp);
    size_t n = hn::CompressStore(indice_vec, mask, d, indices.data() + indice_offset + 1);
    indices[indice_offset] = static_cast<uint8_t>(n);
    indice_offset += (n + 1);
  }
  indices.resize(indice_offset);

  indice_offset = 0;
  size_t last_pos = 0;
  size_t vec_idx = 0;
  std::vector<std::string_view> ss;
  while (indice_offset < indices.size()) {
    size_t n = indices[indice_offset];
    for (size_t i = 0; i < n; i++) {
      auto pos = indices[indice_offset + i + 1] + N * vec_idx;
      if (pos != last_pos) {
        auto part = s.substr(last_pos, pos - last_pos);
        ss.emplace_back(part);
      }
      last_pos = pos + 1;
    }
    vec_idx++;
    indice_offset += (n + 1);
  }
  return ss;
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
  //   HWY_EXPORT_T(Table, simd_string_split_by_char_impl1);
  //   return HWY_DYNAMIC_DISPATCH_T(Table)(s, ch);
}

}  // namespace functions
}  // namespace rapidudf
#endif  // HWY_ONCE