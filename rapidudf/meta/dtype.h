/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <string.h>
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

#include <fmt/format.h>
#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"

#include "rapidudf/context/context.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/types/json_object.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

enum CollectionType {
  COLLECTION_INVALID = 0,
  COLLECTION_VECTOR,
  COLLECTION_MAP,
  COLLECTION_SET,
  COLLECTION_UNORDERED_MAP,
  COLLECTION_UNORDERED_SET,
  COLLECTION_ABSL_SPAN,
  COLLECTION_TUPLE,
  COLLECTION_SIMD_VECTOR,
  COLLECTION_END = 64,
};

constexpr std::array<std::string_view, COLLECTION_SIMD_VECTOR + 1> kCollectionTypeStrs = {
    "", "vector", "map", "set", "unordered_map", "unordered_set", "absl_span", "tuple", "simd_vector"};
enum FundamentalType {
  DATA_INVALID = 0,
  DATA_VOID,
  DATA_BIT,
  DATA_U8,
  DATA_I8,
  DATA_U16,
  DATA_I16,
  DATA_U32,
  DATA_I32,
  DATA_U64,
  DATA_I64,
  DATA_F32,
  DATA_F64,
  DATA_STD_STRING_VIEW,
  DATA_STRING_VIEW,
  DATA_STRING,
  DATA_FLATBUFFERS_STRING,
  DATA_JSON,
  DATA_CONTEXT,

  DATA_OBJECT_BEGIN = 64,
  DATA_COMPLEX_OBJECT = (1 << 14) - 1,
};

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

constexpr std::array<std::string_view, DATA_CONTEXT + 1> kFundamentalTypeStrs = {
    "invalid",     "void",   "bit",        "u8",   "i8",     "u16", "i16",
    "u32",         "i32",    "u64",        "i64",  "f32",    "f64", "std_string_view",
    "string_view", "string", "fbs_string", "json", "Context"};

class DType {
 public:
  DType(uint64_t control = 0) : control_(control) {}
  DType(FundamentalType t0, FundamentalType t1 = DATA_INVALID, FundamentalType t2 = DATA_INVALID,
        FundamentalType t3 = DATA_INVALID)
      : control_(0) {
    t0_ = t0;
    t1_ = t1;
    t2_ = t2;
    t3_ = t3;
  }
  DType(const DType& other);
  void SetElement(size_t idx, DType&& dtype);
  uint64_t Control() const;
  FundamentalType GetFundamentalType() const { return static_cast<FundamentalType>(t0_); }
  bool IsSimdVector() const { return container_type_ == COLLECTION_SIMD_VECTOR; }
  bool IsVector() const { return container_type_ == COLLECTION_VECTOR; }
  bool IsAbslSpan() const { return container_type_ == COLLECTION_ABSL_SPAN; }
  bool IsTuple() const { return container_type_ == COLLECTION_TUPLE; }
  bool IsMap() const { return container_type_ == COLLECTION_MAP; }
  bool IsUnorderedMap() const { return container_type_ == COLLECTION_UNORDERED_MAP; }
  bool IsSet() const { return container_type_ == COLLECTION_SET; }
  bool IsCollection() const { return container_type_ != 0; }
  bool IsPtr() const { return ptr_bit_ == 1; }
  bool IsIntegerPtr() const { return IsPtr() && (PtrTo().IsInteger()); }
  bool IsSimdVectorBit() const { return container_type_ == COLLECTION_SIMD_VECTOR && t0_ == DATA_BIT; }
  bool IsPrimitive() const { return IsFundamental() && (t0_ >= DATA_U8 && t0_ <= DATA_STRING_VIEW); }
  bool IsFundamental() const { return ptr_bit_ == 0 && container_type_ == 0; }
  bool IsNumber() const { return IsFundamental() && (t0_ >= DATA_U8 && t0_ <= DATA_F64); }
  bool IsF32() const { return IsFundamental() && t0_ == DATA_F32; }
  bool IsF64() const { return IsFundamental() && t0_ == DATA_F64; }
  bool IsFloat() const { return IsF32() || IsF64(); }
  bool IsInteger() const { return IsFundamental() && (t0_ >= DATA_U8 && t0_ <= DATA_I64); }
  bool IsSigned() const;
  bool IsVoid() const { return t0_ == DATA_VOID; }
  bool IsBit() const { return IsFundamental() && t0_ == DATA_BIT; }
  bool IsBool() const { return IsBit(); }
  bool IsStringView() const { return IsFundamental() && t0_ == DATA_STRING_VIEW; }
  bool IsStdStringView() const { return IsFundamental() && t0_ == DATA_STD_STRING_VIEW; }
  bool IsString() const { return IsFundamental() && t0_ == DATA_STRING; }
  bool IsFlatbuffersString() const { return IsFundamental() && t0_ == DATA_FLATBUFFERS_STRING; }
  bool IsJson() const { return IsFundamental() && t0_ == DATA_JSON; }
  bool IsJsonPtr() const { return IsPtr() && (PtrTo().IsJson()); }
  bool IsVectorPtr() const { return IsPtr() && (PtrTo().IsVector()); }
  bool IsMapPtr() const { return IsPtr() && (PtrTo().IsMap()); }
  bool IsUnorderedMapPtr() const { return IsPtr() && (PtrTo().IsUnorderedMap()); }
  bool IsContext() const { return IsFundamental() && t0_ == DATA_CONTEXT; }
  bool IsContextPtr() const { return IsPtr() && (PtrTo().IsContext()); }
  bool IsStringPtr() const { return IsPtr() && (PtrTo().IsString()); }
  bool IsInvalid() const { return Control() == 0; }
  bool IsComplexObj() const;
  bool IsFlatbuffersStringPtr() const { return IsPtr() && (PtrTo().IsFlatbuffersString()); }
  bool CanCastTo(DType other) const;

  DType Key() const;
  DType Elem() const;
  DType PtrTo() const;
  DType ToSimdVector() const;
  DType ToPtr() const;
  DType ToVector() const;
  DType ToAbslSpan() const;
  DType ToCollection(CollectionType t) const;
  uint32_t TupleSize() const;
  std::vector<DType> ExtractTupleDtypes() const;

  bool IsSameFundamentalType(const DType& other) const { return t0_ == other.t0_; }
  uint32_t ByteSize() const;
  uint32_t Bits() const { return ByteSize() * 8; }
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
  union {
    struct {
      uint64_t container_type_ : kContainerTypeBits;
      uint64_t ptr_bit_ : 1;
      uint64_t t0_ : kPrimitiveTypeBits;
      uint64_t t1_ : kPrimitiveTypeBits;
      uint64_t t2_ : kPrimitiveTypeBits;
      uint64_t t3_ : kPrimitiveTypeBits;
      uint64_t reserved_ : 64 - kContainerTypeBits - 4 * kPrimitiveTypeBits - 1;
    };
    uint64_t control_;
  };
  std::vector<std::shared_ptr<DType>> element_types_;
};
// static_assert(sizeof(DType) == 8, "sizeof(DType) != 8");

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
                is_specialization<xtype, simd::Vector>::value) {                                                      \
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
    // if constexpr (std::is_pointer<Origin>::value) {
    //   static_assert(sizeof(Origin) == -1, "Can NOT get dtype for ptr of ptr");
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
  if constexpr (is_specialization<T, simd::Vector>::value) {
    using val_type = typename T::value_type;
    RETURN_IF_NOT_FUNDAMENTAL_TYPE(val_type)
    auto v = get_dtype<typename T::value_type>();
    return v.ToCollection(COLLECTION_SIMD_VECTOR);
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
    if constexpr (std::tuple_size_v<T> > 4) {
      static_assert(sizeof(T) == -1, "Too many tuple args for dtype, max 4 args supported.");
    }
    DType d0 = get_dtype<typename std::tuple_element<0, T>::type>();
    DType d1, d2, d3;
    if constexpr (std::tuple_size_v<T> > 1) {
      d1 = get_dtype<typename std::tuple_element<1, T>::type>();
    }
    if constexpr (std::tuple_size_v<T> > 2) {
      d2 = get_dtype<typename std::tuple_element<2, T>::type>();
    }
    if constexpr (std::tuple_size_v<T> > 3) {
      d3 = get_dtype<typename std::tuple_element<3, T>::type>();
    }

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
    if (d2.IsPtr() || d2.IsCollection()) {
      t2 = DATA_COMPLEX_OBJECT;
    } else {
      t2 = d2.GetFundamentalType();
    }
    if (d3.IsPtr() || d3.IsCollection()) {
      t3 = DATA_COMPLEX_OBJECT;
    } else {
      t3 = d3.GetFundamentalType();
    }
    DType v(t0, t1, t2, t3);
    if (t0 == DATA_COMPLEX_OBJECT) {
      v.SetElement(0, std::move(d0));
    }
    if (t1 == DATA_COMPLEX_OBJECT) {
      v.SetElement(1, std::move(d1));
    }
    if (t2 == DATA_COMPLEX_OBJECT) {
      v.SetElement(1, std::move(d2));
    }
    if (t3 == DATA_COMPLEX_OBJECT) {
      v.SetElement(1, std::move(d3));
    }
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
    if (c <= rapidudf::DATA_JSON) {
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
