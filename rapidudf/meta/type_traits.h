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
#include <array>
#include <functional>
#include <tuple>
#include <type_traits>

namespace rapidudf {
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};
template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template <typename T>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

template <typename F>
struct FunctionTraits;

template <typename R, typename... Args>
struct FunctionTraits<R (*)(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename R, typename... Args>
struct FunctionTraits<R(Args...)> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename R, typename... Args>
struct FunctionTraits<std::function<R(Args...)>> {
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename C, typename R, typename... Args>
struct FunctionTraits<R (C::*)(Args...)> {
  using object_type = C;
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename C, typename R, typename... Args>
struct FunctionTraits<R (C::*)(Args...) const> {
  using object_type = C;
  using return_type = R;
  using arguments = std::tuple<Args...>;
};

template <typename F, std::size_t N>
using function_argument_type_t = typename std::tuple_element<N, typename FunctionTraits<F>::arguments>::type;

template <typename F>
using first_function_argument_type_t = function_argument_type_t<F, 0>;
}  // namespace rapidudf