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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

namespace rapidudf {
namespace detail {

// Implemented this way because of a bug in Clang for ARMv7, which gives the
// wrong result for `alignof` a `union` with a field of each scalar type.
template <typename... Ts>
struct max_align_t_ {
  static constexpr std::size_t value() {
    std::size_t const values[] = {0u, alignof(Ts)...};
    std::size_t r = 0u;
    for (auto const v : values) {
      r = r < v ? v : r;
    }
    return r;
  }
};
using max_align_v_ = max_align_t_<long double, double, float, long long int, long int, int, short int, bool, char,
                                  char16_t, char32_t, wchar_t, void*, std::max_align_t>;

}  // namespace detail

template <typename T>
T align_to(T n, T align) {
  return (n + align - 1) & ~(align - 1);
}

/// valid_align_value
///
/// Returns whether an alignment value is valid. Valid alignment values are
/// powers of two representable as std::uintptr_t, with possibly additional
/// context-specific restrictions that are not checked here.
struct valid_align_value_fn {
  static_assert(sizeof(std::size_t) <= sizeof(std::uintptr_t));
  constexpr bool operator()(std::size_t align) const noexcept { return align && !(align & (align - 1)); }
  constexpr bool operator()(std::align_val_t align) const noexcept {
    return operator()(static_cast<std::size_t>(align));
  }
};
inline constexpr valid_align_value_fn valid_align_value;

constexpr std::size_t max_align_v = detail::max_align_v_::value();

/// align_ceil
/// align_ceil_fn
///
/// Returns pointer rounded up to the given alignment.
struct align_ceil_fn {
  constexpr std::uintptr_t operator()(std::uintptr_t x, std::size_t alignment) const {
    assert(valid_align_value(alignment));
    auto alignmentAsInt = static_cast<std::intptr_t>(alignment);
    return (x + alignmentAsInt - 1) & (-alignmentAsInt);
  }

  template <typename T>
  T* operator()(T* x, std::size_t alignment) const {
    auto asUint = reinterpret_cast<std::uintptr_t>(x);
    asUint = (*this)(asUint, alignment);
    return reinterpret_cast<T*>(asUint);
  }
};
inline constexpr align_ceil_fn align_ceil;

}  // namespace rapidudf