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
#include <array>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>
#include "absl/types/span.h"
#include "rapidudf/types/string_view.h"

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

// 基础模板：用于处理至少有一个参数的情况
template <typename First, typename... Rest>
struct FirstOfVariadic {
  using type = First;
};

// 辅助类型别名，使代码更简洁
template <typename... Args>
using first_of_variadic_t = typename FirstOfVariadic<Args...>::type;

template <typename... T>
struct ConstPointerFunctionSignatureHelper;
template <typename T0, typename T1, typename... Ts>
struct ConstPointerFunctionSignatureHelper<T0, T1, Ts...> {
  using type = std::function<void(const T0*, const T1*, const Ts*...)>;
};

template <typename T0>
struct ConstPointerFunctionSignatureHelper<T0> {
  using type = std::function<void(const T0*)>;
};

template <>
struct ConstPointerFunctionSignatureHelper<> {
  using type = std::function<void()>;
};

template <typename T, typename = std::void_t<>>
struct is_destructor_disabled : std::false_type {};

template <typename T>
struct is_destructor_disabled<T, std::void_t<decltype(T::destructor_disabled)>>
    : std::conditional_t<T::destructor_disabled, std::true_type, std::false_type> {};
template <typename T>
struct is_destructor_disabled<T, std::enable_if_t<std::is_trivially_destructible_v<T>>> : std::true_type {};

// 辅助类型别名，简化使用
template <typename T>
constexpr bool is_destructor_disabled_v = is_destructor_disabled<T>::value;

}  // namespace rapidudf