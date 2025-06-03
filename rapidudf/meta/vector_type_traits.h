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

template <typename CType>
struct VectorTypeTraits {
  using ElementType = CType;
};

template <typename CType>
struct VectorTypeTraits<std::vector<CType>> {
  using ElementType = absl::Span<const CType>;
};

template <typename CType>
struct VectorTypeTraits<absl::Span<CType>> {
  using ElementType = absl::Span<const CType>;
};

template <typename CType, size_t N>
struct VectorTypeTraits<std::array<CType, N>> {
  using ElementType = absl::Span<const CType>;
};

template <typename KeyType, typename ValueType>
struct VectorTypeTraits<std::map<KeyType, ValueType>> {
  using ElementType = std::pair<absl::Span<const KeyType>, absl::Span<const ValueType>>;
};

template <size_t N>
struct VectorTypeTraits<std::array<std::string, N>> {
  using ElementType = absl::Span<const StringView>;
};
template <size_t N>
struct VectorTypeTraits<std::array<std::string_view, N>> {
  using ElementType = absl::Span<const StringView>;
};

template <>
struct VectorTypeTraits<std::string> {
  using DataType = StringView;
};
template <>
struct VectorTypeTraits<std::string_view> {
  using DataType = StringView;
};

template <>
struct VectorTypeTraits<std::vector<std::string>> {
  using ElementType = absl::Span<const StringView>;
};
template <>
struct VectorTypeTraits<std::vector<std::string_view>> {
  using ElementType = absl::Span<const StringView>;
};

}  // namespace rapidudf