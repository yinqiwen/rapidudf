/*
 * Copyright (c) 2025 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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

#include <arrow/array/builder_primitive.h>
#include <arrow/memory_pool.h>
#include <arrow/scalar.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <array>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "arrow/api.h"
#include "arrow/type_traits.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/types/bit.h"

namespace arrow {

template <>
struct CTypeTraits<rapidudf::Bit> : public TypeTraits<BooleanType> {
  using ArrowType = BooleanType;
};

template <typename CType>
struct CTypeTraits<CType*> : public TypeTraits<UInt64Type> {
  using ArrowType = UInt64Type;
};
template <typename CType>
struct CTypeTraits<const CType*> : public TypeTraits<UInt64Type> {
  using ArrowType = UInt64Type;
};

template <typename CType>
struct CTypeTraits<std::set<CType>> : public TypeTraits<ListType> {
  using ArrowType = ListType;

  static inline std::shared_ptr<DataType> type_singleton() { return list(CTypeTraits<CType>::type_singleton()); }
};
template <typename CType>
struct CTypeTraits<std::list<CType>> : public TypeTraits<ListType> {
  using ArrowType = ListType;

  static inline std::shared_ptr<DataType> type_singleton() { return list(CTypeTraits<CType>::type_singleton()); }
};
template <typename CType>
struct CTypeTraits<std::unordered_set<CType>> : public TypeTraits<ListType> {
  using ArrowType = ListType;

  static inline std::shared_ptr<DataType> type_singleton() { return list(CTypeTraits<CType>::type_singleton()); }
};
template <typename KCType, typename VCType>
struct CTypeTraits<std::map<KCType, VCType>> : public TypeTraits<MapType> {
  using ArrowType = MapType;
  static inline std::shared_ptr<DataType> type_singleton() {
    return map(CTypeTraits<KCType>::type_singleton(), CTypeTraits<VCType>::type_singleton());
  }
};
template <typename KCType, typename VCType>
struct CTypeTraits<std::unordered_map<KCType, VCType>> : public TypeTraits<MapType> {
  using ArrowType = MapType;
  static inline std::shared_ptr<DataType> type_singleton() {
    return map(CTypeTraits<KCType>::type_singleton(), CTypeTraits<VCType>::type_singleton());
  }
};

}  // namespace arrow

namespace rapidudf {
template <typename T>
std::shared_ptr<arrow::DataType> get_arrow_dtype() {
  if (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
    return arrow::binary_view();
  }
  return arrow::CTypeTraits<T>::type_singleton();
}

template <typename CType>
struct VectorElementTypeTraits {
  using DataType = CType;
};

template <>
struct VectorElementTypeTraits<std::string> {
  using DataType = std::string_view;
};

// template <typename T>
// std::shared_ptr<arrow::DataType> get_arrow_dtype() {
//   if constexpr (std::is_const_v<T>) {
//     using Origin = std::remove_const_t<T>;
//     return get_arrow_dtype<Origin>();
//   }
//   if constexpr (std::is_pointer<T>::value) {
//     return get_arrow_dtype<uint64_t>();
//   }
//   if constexpr (std::is_reference_v<T>) {
//     return get_arrow_dtype<uint64_t>();
//   }

//   if constexpr (std::is_same_v<bool, T>) {
//     return arrow::boolean();
//   }
//   if constexpr (std::is_same_v<char, T>) {
//     return arrow::int8();
//   }
//   if constexpr (std::is_same_v<int8_t, T>) {
//     return arrow::int8();
//   }
//   if constexpr (std::is_same_v<uint8_t, T>) {
//     return arrow::uint8();
//   }
//   if constexpr (std::is_same_v<int16_t, T>) {
//     return arrow::int16();
//   }
//   if constexpr (std::is_same_v<uint16_t, T>) {
//     return arrow::uint16();
//   }
//   if constexpr (std::is_same_v<int32_t, T>) {
//     return arrow::int32();
//   }
//   if constexpr (std::is_same_v<uint32_t, T>) {
//     return arrow::uint32();
//   }
//   if constexpr (std::is_same_v<int64_t, T>) {
//     return arrow::int64();
//   }
//   if constexpr (std::is_same_v<uint64_t, T>) {
//     return arrow::uint64();
//   }
//   if constexpr (std::is_same_v<size_t, T>) {
//     return arrow::uint64();
//   }
//   if constexpr (std::is_same_v<float, T>) {
//     return arrow::float32();
//   }
//   if constexpr (std::is_same_v<double, T>) {
//     return arrow::float64();
//   }
//   if constexpr (std::is_same_v<std::string_view, T>) {
//     return arrow::binary_view();
//   }
//   if constexpr (std::is_same_v<std::string, T>) {
//     return arrow::binary_view();
//   }
//   if constexpr (std::is_same_v<Bit, T>) {
//     return arrow::boolean();
//   }
//   if constexpr (is_specialization<T, std::vector>::value) {
//     auto v = get_arrow_dtype<typename T::value_type>();
//     if (!v) {
//       return nullptr;
//     } else {
//       return arrow::list(v);
//     }
//   }
//   if constexpr (is_std_array_v<T>) {
//     auto v = get_arrow_dtype<typename T::value_type>();
//     return arrow::fixed_size_list(v, std::tuple_size<T>::value);
//   }
//   if constexpr (is_specialization<T, std::vector>::value || is_specialization<T, std::set>::value ||
//                 is_specialization<T, std::unordered_set>::value || is_specialization<T, std::list>::value) {
//     auto v = get_arrow_dtype<typename T::value_type>();
//     return arrow::list(v);
//   }

//   if constexpr (is_specialization<T, std::map>::value || is_specialization<T, std::unordered_map>::value) {
//     using key_type = typename T::key_type;
//     using val_type = typename T::mapped_type;
//     auto k = get_arrow_dtype<key_type>();
//     auto v = get_arrow_dtype<val_type>();
//     return arrow::map(k, v);
//   }

//   return nullptr;
// }

inline arrow::MemoryPool* default_arrow_memory_pool() { return arrow::default_memory_pool(); }

}  // namespace rapidudf