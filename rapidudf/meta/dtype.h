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

#include <string.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"
#include "fmt/format.h"

#include "rapidudf/context/context.h"
#include "rapidudf/meta//dtype_enums.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/json_object.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {

struct DTypeAttr {
  std::string schema;  // for 'dyn_obj'
};

class DType {
 public:
  DType(uint64_t control = 0) { ctrl_.control_ = control; }
  DType(FundamentalType t0, FundamentalType t1 = DATA_INVALID) {
    ctrl_.control_ = 0;
    ctrl_.t0_ = t0;
    ctrl_.t1_ = t1;
    // ctrl_.t2_ = t2;
    // ctrl_.t3_ = t3;
  }
  DType(const DType& other);
  void SetElement(size_t idx, DType&& dtype);
  uint64_t Control() const;
  void Reset();
  FundamentalType GetFundamentalType() const { return static_cast<FundamentalType>(ctrl_.t0_); }
  bool IsSimdVector(size_t fixed_size = 0) const {
    if (fixed_size == 0) {
      return ctrl_.container_type_ == COLLECTION_SIMD_VECTOR;
    } else {
      return ctrl_.container_type_ == COLLECTION_SIMD_VECTOR && fixed_size == ctrl_.fixed_array_size_;
    }
  }
  bool IsSimdVectorPtr(size_t fixed_size = 0) const { return IsPtr() && (PtrTo().IsSimdVector(fixed_size)); }
  bool IsSimdVectorBitPtr() const { return IsSimdVectorPtr() && ctrl_.t0_ == DATA_BIT; }

  bool IsVector() const { return ctrl_.container_type_ == COLLECTION_VECTOR; }
  bool IsArray(size_t fixed_size = 0) const {
    if (fixed_size == 0) {
      return ctrl_.container_type_ == COLLECTION_ARRAY;
    } else {
      return ctrl_.container_type_ == COLLECTION_ARRAY && fixed_size == ctrl_.fixed_array_size_;
    }
  }
  bool IsAbslSpan() const { return ctrl_.container_type_ == COLLECTION_ABSL_SPAN; }
  bool IsTuple() const { return ctrl_.container_type_ == COLLECTION_TUPLE; }
  bool IsMap() const { return ctrl_.container_type_ == COLLECTION_MAP; }
  bool IsUnorderedMap() const { return ctrl_.container_type_ == COLLECTION_UNORDERED_MAP; }
  bool IsSet() const { return ctrl_.container_type_ == COLLECTION_SET; }
  bool IsCollection() const { return ctrl_.container_type_ != 0; }
  bool IsPtr() const { return ctrl_.ptr_bit_ == 1; }
  bool IsIntegerPtr() const { return IsPtr() && (PtrTo().IsInteger()); }

  bool IsPrimitive() const { return IsFundamental() && (ctrl_.t0_ >= DATA_BIT && ctrl_.t0_ <= DATA_STRING_VIEW); }
  bool IsFundamental() const { return ctrl_.ptr_bit_ == 0 && ctrl_.container_type_ == 0; }
  bool IsNumber() const { return IsFundamental() && (ctrl_.t0_ >= DATA_U8 && ctrl_.t0_ <= DATA_F80); }
  bool IsF16() const { return IsFundamental() && ctrl_.t0_ == DATA_F16; }
  bool IsF32() const { return IsFundamental() && ctrl_.t0_ == DATA_F32; }
  bool IsF64() const { return IsFundamental() && ctrl_.t0_ == DATA_F64; }
  bool IsF80() const { return IsFundamental() && ctrl_.t0_ == DATA_F80; }
  bool IsFloat() const { return IsF16() || IsF32() || IsF64() || IsF80(); }
  bool IsInteger() const { return IsFundamental() && (ctrl_.t0_ >= DATA_U8 && ctrl_.t0_ <= DATA_I64); }
  bool IsI64() const { return IsFundamental() && (ctrl_.t0_ == DATA_I64); }
  bool IsU64() const { return IsFundamental() && (ctrl_.t0_ == DATA_U64); }
  bool IsI32() const { return IsFundamental() && (ctrl_.t0_ == DATA_I32); }
  bool IsU32() const { return IsFundamental() && (ctrl_.t0_ == DATA_U32); }
  bool IsSigned() const;
  bool IsVoid() const { return ctrl_.t0_ == DATA_VOID; }
  bool IsVoidPtr() const { return IsVoid() && IsPtr(); }
  bool IsBit() const { return IsFundamental() && ctrl_.t0_ == DATA_BIT; }
  bool IsBool() const { return IsBit(); }
  bool IsStringView() const { return IsFundamental() && ctrl_.t0_ == DATA_STRING_VIEW; }
  bool IsStdStringView() const { return IsFundamental() && ctrl_.t0_ == DATA_STD_STRING_VIEW; }
  bool IsString() const { return IsFundamental() && ctrl_.t0_ == DATA_STRING; }
  bool IsFlatbuffersString() const { return IsFundamental() && ctrl_.t0_ == DATA_FLATBUFFERS_STRING; }
  bool IsJson() const { return IsFundamental() && ctrl_.t0_ == DATA_JSON; }
  bool IsJsonPtr() const { return IsPtr() && (PtrTo().IsJson()); }
  bool IsVectorPtr() const { return IsPtr() && (PtrTo().IsVector()); }
  bool IsMapPtr() const { return IsPtr() && (PtrTo().IsMap()); }
  bool IsUnorderedMapPtr() const { return IsPtr() && (PtrTo().IsUnorderedMap()); }
  bool IsContext() const { return IsFundamental() && ctrl_.t0_ == DATA_CONTEXT; }
  bool IsContextPtr() const { return IsPtr() && (PtrTo().IsContext()); }
  bool IsStringPtr() const { return IsPtr() && (PtrTo().IsString()); }
  bool IsDynObject() const { return IsFundamental() && ctrl_.t0_ == DATA_DYN_OBJECT; }
  bool IsDynObjectPtr() const { return IsPtr() && (PtrTo().IsDynObject()); }
  bool IsInvalid() const { return Control() == 0; }
  bool IsComplexObj() const;
  bool IsFlatbuffersStringPtr() const { return IsPtr() && (PtrTo().IsFlatbuffersString()); }
  bool CanCastTo(DType other) const;

  size_t Hash(const uint8_t* data) const;
  bool Equal(const uint8_t* left, const uint8_t* right) const;

  DType Key() const;
  DType Elem() const;
  DType PtrTo() const;
  DType ToSimdVector() const;
  DType ToPtr() const;
  DType ToVector() const;
  DType ToAbslSpan() const;
  DType ToCollection(CollectionType t) const;
  DType ToArray(size_t fixed_size) const;
  uint32_t TupleSize() const;
  std::vector<DType> ExtractTupleDtypes() const;

  bool IsSameFundamentalType(const DType& other) const { return ctrl_.t0_ == other.ctrl_.t0_; }
  uint32_t ByteSize() const;
  uint32_t Bits() const {
    if (IsBit()) {
      return 1;
    }
    return ByteSize() * 8;
  }
  uint32_t QwordSize() const {
    uint32_t n = ByteSize() / 8;
    if (ByteSize() % 8 > 0) {
      n++;
    }
    return n;
  }

  DType& operator=(const DType& other);
  bool operator==(const DType& other) const;
  bool operator!=(const DType& other) const;
  bool operator>(const DType& other) const;
  bool operator<(const DType& other) const;
  bool operator>=(const DType& other) const;
  bool operator<=(const DType& other) const;
  int Compare(const DType& other) const;

  template <typename T>
  std::optional<T> ToPrimitiveValue(uint64_t bin);
  template <typename T>
  std::optional<uint64_t> FromPrimitiveValue(T v);

  std::string ToString() const;
  std::string GetTypeString() const;

  static std::string Demangle(const char* name);

 private:
  static constexpr uint32_t kContainerTypeBits = 6;
  static constexpr uint32_t kPrimitiveTypeBits = 14;
  static constexpr uint32_t kFixedArraySizeBits = 29;

  union ControlValue {
    struct {
      uint64_t container_type_ : kContainerTypeBits;
      uint64_t ptr_bit_ : 1;
      uint64_t t0_ : kPrimitiveTypeBits;
      uint64_t t1_ : kPrimitiveTypeBits;
      uint64_t fixed_array_size_ : kFixedArraySizeBits;
      // uint64_t reserved_ : 64 - kContainerTypeBits - 2 * kPrimitiveTypeBits - kFixedVectorSizeBits - 1;
    };
    uint64_t control_;
    ControlValue(uint64_t v = 0) { control_ = v; }
  } ctrl_;
  std::shared_ptr<std::vector<std::shared_ptr<DType>>> element_types_;
};
static_assert(sizeof(DType) == 8 + sizeof(std::shared_ptr<int>), "sizeof(DType) != 8");

template <typename T>
DType get_dtype();

class DTypeFactory {
 public:
  static DType GetDTypeByName(const std::string& name);
  static std::string_view GetNameByDType(DType dtype);
  static void Visit(std::function<void(const std::string&, DType)>&& f);

  template <typename T>
  static bool Add(const std::string& name) {
    DType dtype = get_dtype<T>();
    if (dtype.IsPtr()) {
      return false;
    }
    return AddNameDType(name, dtype);
  }

  template <typename T>
  static bool Add() {
    DType dtype = get_dtype<T>();
    if (dtype.IsPtr()) {
      return false;
    }
    std::string name;
    if (dtype.IsCollection()) {
      name = dtype.GetTypeString();
    } else {
      name = DType::Demangle(typeid(T).name());
    }
    return AddNameDType(name, dtype);
  }

 private:
  static bool AddNameDType(const std::string& name, DType dtype);

  template <typename T>
  friend DType get_dtype();
};

static inline uint32_t nextTypeId() {
  static std::atomic<uint32_t> type_id_seed = {DATA_OBJECT_BEGIN};
  return type_id_seed.fetch_add(1);
}

#define RETURN_IF_NOT_FUNDAMENTAL_TYPE(xtype)                                                                         \
  if constexpr (std::is_pointer<xtype>::value || is_specialization<xtype, std::vector>::value ||                      \
                is_specialization<xtype, std::set>::value || is_specialization<xtype, std::map>::value ||             \
                is_specialization<xtype, std::unordered_map>::value ||                                                \
                is_specialization<xtype, std::unordered_set>::value || is_specialization<xtype, absl::Span>::value || \
                is_specialization<xtype, std::pair>::value || is_specialization<xtype, std::tuple>::value ||          \
                is_specialization<xtype, Vector>::value) {                                                            \
    static_assert(sizeof(xtype) == -1, "Can NOT get dtype for complex type");                                         \
    return {};                                                                                                        \
  }

template <typename T>
DType get_dtype() {
  if constexpr (std::is_void_v<T>) {
    return DType(DATA_VOID);
  }
  if constexpr (std::is_const_v<T>) {
    using Origin = std::remove_const_t<T>;
    return get_dtype<Origin>();
  }

  if constexpr (std::is_pointer<T>::value) {
    using Origin = std::remove_pointer_t<T>;
    auto v = get_dtype<Origin>();
    if (v.IsPtr()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToPtr();
      complex.SetElement(0, std::move(v));
      return complex;
    } else {
      return v.ToPtr();
    }
  }
  if constexpr (std::is_reference_v<T>) {
    using Origin = std::remove_reference_t<T>;
    // if constexpr (std::is_pointer<Origin>::value) {
    //   static_assert(sizeof(Origin) == -1, "Can NOT get dtype");
    //   return {};
    // }
    auto v = get_dtype<Origin>();
    if (v.IsPtr()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToPtr();
      complex.SetElement(0, std::move(v));
      return complex;
    } else {
      return v.ToPtr();
    }
  }
  if constexpr (is_specialization<T, Vector>::value) {
    // using val_type = typename T::value_type;
    // if constexpr (is_specialization<val_type, simd::Vector>::value || ){

    // }
    // RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<typename T::value_type>();
    if (v.IsPtr() || v.IsCollection()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToCollection(COLLECTION_SIMD_VECTOR);
      complex.SetElement(0, std::move(v));
      return complex;
    } else {
      return v.ToCollection(COLLECTION_SIMD_VECTOR);
    }
  }
  if constexpr (is_specialization<T, std::vector>::value) {
    using val_type = typename T::value_type;
    // RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<val_type>();
    if (v.IsPtr() || v.IsCollection()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToVector();
      complex.SetElement(0, std::move(v));
      return complex;
    } else {
      return v.ToCollection(COLLECTION_VECTOR);
    }
  }
  if constexpr (is_std_array_v<T>) {
    using val_type = typename T::value_type;
    RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<val_type>();
    return v.ToArray(std::tuple_size<T>::value);
  }
  if constexpr (is_specialization<T, std::set>::value || is_specialization<T, std::unordered_set>::value) {
    using val_type = typename T::value_type;
    // RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<val_type>();
    CollectionType t;
    if constexpr (is_specialization<T, std::set>::value) {
      t = COLLECTION_SET;
    } else {
      t = COLLECTION_UNORDERED_SET;
    }
    if (v.IsPtr() || v.IsCollection()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToCollection(t);
      complex.SetElement(0, std::move(v));
      return complex;
    } else {
      return v.ToCollection(t);
    }
  }
  if constexpr (is_specialization<T, absl::Span>::value) {
    using val_type = typename T::value_type;
    // RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<val_type>();
    if (v.IsPtr() || v.IsCollection()) {
      DType complex(DATA_COMPLEX_OBJECT);
      complex = complex.ToCollection(COLLECTION_ABSL_SPAN);
      complex.SetElement(0, std ::move(v));
      return complex;
    } else {
      return v.ToCollection(COLLECTION_ABSL_SPAN);
    }
  }
  if constexpr (is_specialization<T, std::map>::value || is_specialization<T, std::unordered_map>::value ||
                is_specialization<T, std::pair>::value) {
    DType key_v, value_v;
    if constexpr (is_specialization<T, std::map>::value || is_specialization<T, std::unordered_map>::value) {
      using key_type = typename T::key_type;
      using val_type = typename T::mapped_type;
      key_v = get_dtype<key_type>();
      value_v = get_dtype<val_type>();
    } else {
      using first_type = typename T::first_type;
      using second_type = typename T::second_type;
      key_v = get_dtype<first_type>();
      value_v = get_dtype<second_type>();
    }

    CollectionType t;
    if constexpr (is_specialization<T, std::map>::value) {
      t = COLLECTION_MAP;
    } else if constexpr (is_specialization<T, std::unordered_map>::value) {
      t = COLLECTION_UNORDERED_MAP;
    } else {
      t = COLLECTION_TUPLE;
    }

    FundamentalType key_t, value_t;
    if (key_v.IsPtr() || key_v.IsCollection()) {
      key_t = DATA_COMPLEX_OBJECT;
    } else {
      key_t = key_v.GetFundamentalType();
    }
    if (value_v.IsPtr() || value_v.IsCollection()) {
      value_t = DATA_COMPLEX_OBJECT;
    } else {
      value_t = value_v.GetFundamentalType();
    }
    DType v(key_t, value_t);
    if (key_t == DATA_COMPLEX_OBJECT) {
      v.SetElement(0, std::move(key_v));
    }
    if (value_t == DATA_COMPLEX_OBJECT) {
      v.SetElement(1, std::move(value_v));
    }
    return v.ToCollection(t);
  }

  if constexpr (is_specialization<T, std::tuple>::value) {
    if constexpr (std::tuple_size_v<T> > 2) {
      static_assert(sizeof(T) == -1, "Too many tuple args for dtype, max 2 args supported.");
    }
    DType d0 = get_dtype<typename std::tuple_element<0, T>::type>();
    DType d1, d2, d3;
    if constexpr (std::tuple_size_v<T> > 1) {
      d1 = get_dtype<typename std::tuple_element<1, T>::type>();
    }
    // if constexpr (std::tuple_size_v<T> > 2) {
    //   d2 = get_dtype<typename std::tuple_element<2, T>::type>();
    // }
    // if constexpr (std::tuple_size_v<T> > 3) {
    //   d3 = get_dtype<typename std::tuple_element<3, T>::type>();
    // }

    FundamentalType t0, t1, t2, t3;
    if (d0.IsPtr() || d0.IsCollection()) {
      t0 = DATA_COMPLEX_OBJECT;
    } else {
      t0 = d0.GetFundamentalType();
    }
    if (d1.IsPtr() || d1.IsCollection()) {
      t1 = DATA_COMPLEX_OBJECT;
    } else {
      t1 = d1.GetFundamentalType();
    }
    // if (d2.IsPtr() || d2.IsCollection()) {
    //   t2 = DATA_COMPLEX_OBJECT;
    // } else {
    //   t2 = d2.GetFundamentalType();
    // }
    // if (d3.IsPtr() || d3.IsCollection()) {
    //   t3 = DATA_COMPLEX_OBJECT;
    // } else {
    //   t3 = d3.GetFundamentalType();
    // }
    DType v(t0, t1);
    if (t0 == DATA_COMPLEX_OBJECT) {
      v.SetElement(0, std::move(d0));
    }
    if (t1 == DATA_COMPLEX_OBJECT) {
      v.SetElement(1, std::move(d1));
    }
    // if (t2 == DATA_COMPLEX_OBJECT) {
    //   v.SetElement(1, std::move(d2));
    // }
    // if (t3 == DATA_COMPLEX_OBJECT) {
    //   v.SetElement(1, std::move(d3));
    // }
    return v.ToCollection(COLLECTION_TUPLE);
  }

  if constexpr (std::is_same_v<bool, T>) {
    return DType(DATA_BIT);
  }
  if constexpr (std::is_same_v<char, T>) {
    return DType(DATA_I8);
  }
  if constexpr (std::is_same_v<int8_t, T>) {
    return DType(DATA_I8);
  }
  if constexpr (std::is_same_v<uint8_t, T>) {
    return DType(DATA_U8);
  }
  if constexpr (std::is_same_v<int16_t, T>) {
    return DType(DATA_I16);
  }
  if constexpr (std::is_same_v<uint16_t, T>) {
    return DType(DATA_U16);
  }
  if constexpr (std::is_same_v<int32_t, T>) {
    return DType(DATA_I32);
  }
  if constexpr (std::is_same_v<uint32_t, T>) {
    return DType(DATA_U32);
  }
  if constexpr (std::is_same_v<int64_t, T>) {
    return DType(DATA_I64);
  }
  if constexpr (std::is_same_v<uint64_t, T>) {
    return DType(DATA_U64);
  }
  if constexpr (std::is_same_v<size_t, T>) {
    return DType(DATA_U64);
  }
  if constexpr (std::is_same_v<float, T>) {
    return DType(DATA_F32);
  }
  if constexpr (std::is_same_v<double, T>) {
    return DType(DATA_F64);
  }
  if constexpr (std::is_same_v<long double, T>) {
    return DType(DATA_F80);
  }
  if constexpr (std::is_same_v<std::string_view, T>) {
    return DType(DATA_STD_STRING_VIEW);
  }
  if constexpr (std::is_same_v<StringView, T>) {
    return DType(DATA_STRING_VIEW);
  }
  if constexpr (std::is_same_v<std::string, T>) {
    return DType(DATA_STRING);
  }
  if constexpr (std::is_same_v<flatbuffers::String, T>) {
    return DType(DATA_FLATBUFFERS_STRING);
  }
  if constexpr (std::is_same_v<JsonObject, T>) {
    return DType(DATA_JSON);
  }
  if constexpr (std::is_same_v<Bit, T>) {
    return DType(DATA_BIT);
  }
  if constexpr (std::is_same_v<Context, T>) {
    return DType(DATA_CONTEXT);
  }
  if constexpr (std::is_base_of_v<DynObject, T>) {
    return DType(DATA_DYN_OBJECT);
  }

  if constexpr (std::is_same_v<Pointer, T>) {
    return DType(DATA_POINTER);
  }
  static uint32_t id = nextTypeId();
  DType dtype(static_cast<FundamentalType>(id));
  // DTypeFactory::Add<T>(dtype);
  return dtype;
}

template <typename T>
struct get_dtypes_helper {
  static std::vector<DType> get() {
    std::vector<DType> ts{get_dtype<T>()};
    return ts;
  }
};

template <typename... Args>
struct get_dtypes_helper<std::tuple<Args...>> {
  static std::vector<DType> get() {
    std::vector<DType> ts{get_dtype<Args>()...};
    return ts;
  }
};
template <typename L, typename R>
struct get_dtypes_helper<std::pair<L, R>> {
  static std::vector<DType> get() { return std::vector<DType>{get_dtype<L>(), get_dtype<R>()}; }
};

template <typename T>
std::vector<DType> get_dtypes() {
  return get_dtypes_helper<T>::get();
}

uint64_t convert_to(uint64_t val, DType src_dtype, DType dst_type);

template <typename T>
std::optional<T> DType::ToPrimitiveValue(uint64_t bin) {
  if (!IsFundamental()) {
    return {};
  }
  switch (GetFundamentalType()) {
    case DATA_F32: {
      if constexpr (std::is_same_v<float, T>) {
        uint32_t int_val = static_cast<uint32_t>(bin);
        float fv;
        memcpy(&fv, &int_val, sizeof(float));
        return fv;
      } else {
        return {};
      }
    }
    case DATA_F64: {
      if constexpr (std::is_same_v<double, T>) {
        double fv;
        memcpy(&fv, &bin, sizeof(double));
        return fv;
      } else {
        return {};
      }
    }
    case DATA_U64: {
      if constexpr (std::is_same_v<uint64_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_I64: {
      if constexpr (std::is_same_v<int64_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_U32: {
      if constexpr (std::is_same_v<uint32_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_I32: {
      if constexpr (std::is_same_v<int32_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_U16: {
      if constexpr (std::is_same_v<uint16_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_I16: {
      if constexpr (std::is_same_v<int16_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    case DATA_U8: {
      if constexpr (std::is_same_v<uint8_t, T>) {
        return static_cast<T>(bin);
      } else if constexpr (std::is_same_v<bool, T>) {
        return bin > 0 ? true : false;
      } else {
        return {};
      }
    }
    case DATA_I8: {
      if constexpr (std::is_same_v<int8_t, T>) {
        return static_cast<T>(bin);
      } else {
        return {};
      }
    }
    default: {
      return {};
    }
  }
}

template <typename T>
std::optional<uint64_t> DType::FromPrimitiveValue(T val) {
  if constexpr (std::is_same_v<double, T> || std::is_same_v<float, T> || std::numeric_limits<T>::is_integer) {
    if (IsF64()) {
      double dv = static_cast<double>(val);
      uint64_t int_val = 0;
      memcpy(&int_val, &dv, sizeof(double));
      return int_val;
    } else if (IsF32()) {
      float fv = static_cast<float>(val);
      uint32_t int_val = 0;
      memcpy(&int_val, &fv, sizeof(float));
      return int_val;
    } else if (IsNumber()) {
      uint64_t int_val = static_cast<uint64_t>(val);
      return int_val;
    } else {
      return {};
    }
  } else if constexpr (std::is_pointer_v<T>) {
    if (!IsPtr()) {
      return {};
    }
    uint64_t int_val = reinterpret_cast<uint64_t>(val);
    return int_val;
  } else {
    return {};
  }
}

class DTypeMismatchException : public std::logic_error {
 public:
  explicit DTypeMismatchException(DType current, DType expect, const std::string& msg)
      : std::logic_error(fmt::format("expect dtype:{}, but got dtype:{} at {}", expect, current, msg)) {}
};

}  // namespace rapidudf

template <>
struct fmt::formatter<rapidudf::DType> : formatter<std::string> {
  // parse is inherited from formatter<string_view>.
  auto format(rapidudf::DType c, format_context& ctx) const -> format_context::iterator {
    return formatter<std::string>::format(c.ToString(), ctx);
  }
};
template <>
struct fmt::formatter<rapidudf::FundamentalType> : formatter<std::string> {
  // parse is inherited from formatter<string_view>.
  auto format(rapidudf::FundamentalType c, format_context& ctx) const -> format_context::iterator {
    std::string view = "object";
    if (c < rapidudf::DATA_BUILTIN_TYPE_END) {
      view = std::string(rapidudf::kFundamentalTypeStrs[c]);
    } else {
      view = fmt::format("object/{}", static_cast<int>(c));
    }
    return formatter<std::string>::format(view, ctx);
  }
};

namespace std {
template <>
struct hash<::rapidudf::DType> {
  size_t operator()(const ::rapidudf::DType dtype) const { return std::hash<uint64_t>{}(dtype.Control()); }
};
}  // namespace std

#define THROW_DTYPE_MISMATCH_ERR(current, expect)                                                      \
  do {                                                                                                 \
    throw rapidudf::DTypeMismatchException(current, expect, fmt::format("{}:{}", __FILE__, __LINE__)); \
  } while (0)
