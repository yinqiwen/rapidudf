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
#include <type_traits>
#include "rapidudf/reflect/struct.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace reflect {
template <typename T>
struct SimdVectorHelper {
  static T get(simd::Vector<T> v, size_t i) { return v[i]; }
  static size_t size(simd::Vector<T> v) { return v.Size(); }
  static simd::Vector<T> subvector(simd::Vector<T> v, uint32_t pos, uint32_t len) { return v.SubVector(pos, len); }
  static int find(simd::Vector<T> vec, T v) { return vec.Find(v); }
  static int find_neq(simd::Vector<T> vec, T v) { return vec.FindNeq(v); }
  static int find_gt(simd::Vector<T> vec, T v) { return vec.FindGt(v); }
  static int find_ge(simd::Vector<T> vec, T v) { return vec.FindGe(v); }
  static int find_lt(simd::Vector<T> vec, T v) { return vec.FindLt(v); }
  static int find_le(simd::Vector<T> vec, T v) { return vec.FindLe(v); }

  static void Init() {
    RUDF_STRUCT_HELPER_METHODS_BIND(SimdVectorHelper<T>, get, size, subvector);
    if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<StringView, T>) {
      RUDF_STRUCT_HELPER_METHODS_BIND(SimdVectorHelper<T>, find, find_neq, find_gt, find_ge, find_lt, find_le);
    }
  }
};
}  // namespace reflect

}  // namespace rapidudf