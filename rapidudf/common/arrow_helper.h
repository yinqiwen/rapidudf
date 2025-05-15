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

#include <arrow/array/array_primitive.h>
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
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"

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
  if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T> ||
                std::is_same_v<rapidudf::StringView, T>) {
    // return arrow::binary_view();
    return arrow::fixed_size_binary(16);
  } else if constexpr (std::is_same_v<Pointer, T>) {
    return arrow::uint64();
  } else {
    return arrow::CTypeTraits<T>::type_singleton();
  }
}

template <typename CType>
struct VectorElementTypeTraits {
  using DataType = CType;
  using ArrowDataType = CType;
  using BuilderType = typename arrow::CTypeTraits<CType>::BuilderType;
  using ArrayType = typename arrow::CTypeTraits<CType>::ArrayType;
};

template <typename CType>
struct VectorElementTypeTraits<CType*> {
  using DataType = CType*;
  using ArrowDataType = uint64_t;
  using BuilderType = arrow::UInt64Builder;
  using ArrayType = arrow::UInt64Array;
};

template <>
struct VectorElementTypeTraits<std::string> {
  using DataType = StringView;
  using ArrowDataType = StringView;
  using BuilderType = arrow::FixedSizeBinaryBuilder;
  using ArrayType = arrow::FixedSizeBinaryArray;
};

template <>
struct VectorElementTypeTraits<Pointer> {
  using DataType = Pointer;
  using ArrowDataType = uint64_t;
  using BuilderType = arrow::UInt64Builder;
  using ArrayType = arrow::UInt64Array;
};

template <>
struct VectorElementTypeTraits<std::string_view> {
  using DataType = StringView;
  using ArrowDataType = StringView;
  using BuilderType = arrow::FixedSizeBinaryBuilder;
  using ArrayType = arrow::FixedSizeBinaryArray;
};
template <>
struct VectorElementTypeTraits<Bit> {
  using DataType = Bit;
  using ArrowDataType = bool;
  using BuilderType = arrow::BooleanBuilder;
  using ArrayType = arrow::BooleanArray;
};
template <>
struct VectorElementTypeTraits<StringView> {
  using DataType = StringView;
  using ArrowDataType = StringView;
  using BuilderType = arrow::FixedSizeBinaryBuilder;
  using ArrayType = arrow::FixedSizeBinaryArray;
};

inline arrow::MemoryPool* default_arrow_memory_pool() { return arrow::default_memory_pool(); }

}  // namespace rapidudf