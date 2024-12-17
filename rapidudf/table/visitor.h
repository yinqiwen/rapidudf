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

#include <functional>

namespace rapidudf {
namespace table {

enum class VisitStatusCode { kNext = 0, kExit, kReset };

template <typename R, typename... T>
struct VisitorSignatureHelper;
template <typename R, typename T0, typename T1, typename... Ts>
struct VisitorSignatureHelper<R, T0, T1, Ts...> {
  using type = std::function<R(size_t, const T0*, const T1*, const Ts*...)>;
  using return_type = R;
};

template <typename R, typename T0>
struct VisitorSignatureHelper<R, T0> {
  using type = std::function<R(size_t, const T0*)>;
  using return_type = R;
};

template <typename R>
struct VisitorSignatureHelper<R> {
  using type = std::function<R(size_t)>;
  using return_type = R;
};

template <typename... T>
struct MergeVisitorSignatureHelper;

template <typename T0, typename T1, typename... Ts>
struct MergeVisitorSignatureHelper<T0, T1, Ts...> {
  using type = std::function<std::tuple<T0*, T1*, Ts*...>(T0*, T1*, Ts*..., const T0*, const T1*, const Ts*...)>;
};

template <typename T0>
struct MergeVisitorSignatureHelper<T0> {
  using type = std::function<T0*(T0*, const T0*)>;
};

}  // namespace table
}  // namespace rapidudf