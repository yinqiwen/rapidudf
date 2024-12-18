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

#include <tuple>
#include <utility>
namespace rapidudf {

template <typename T>
inline bool is_nullptr(T* ptr) {
  return ptr == nullptr;
}
template <typename Tuple, std::size_t... Is>
inline bool any_nullptr_impl(const Tuple& t, std::index_sequence<Is...>) {
  return (... || is_nullptr(std::get<Is>(t)));
}

// 主函数，用于检查 tuple 中是否有 nullptr
template <typename... Args>
inline bool any_nullptr(const std::tuple<Args...>& t) {
  return any_nullptr_impl(t, std::index_sequence_for<Args...>{});
}

}  // namespace rapidudf