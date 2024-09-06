/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/codegen/dtype.h"
#include <cxxabi.h>
#include "rapidudf/log/log.h"
namespace rapidudf {

static std::unordered_map<std::string, DType>& getNameDTypeMap() {
  static std::unordered_map<std::string, DType> name_to_dtype;
  return name_to_dtype;
}
static std::unordered_map<uint64_t, std::string>& getDTypeNameMap() {
  static std::unordered_map<uint64_t, std::string> dtype_to_name;
  return dtype_to_name;
}

void DTypeFactory::Visit(std::function<void(const std::string&, DType)>&& f) {
  for (const auto& [name, dtype] : getNameDTypeMap()) {
    f(name, dtype);
  }
  for (uint32_t t = DATA_U8; t <= DATA_STRING_VIEW; t++) {
    f(std::string(kFundamentalTypeStrs[t]), static_cast<FundamentalType>(t));
  }
}

std::string_view DTypeFactory::GetNameByDType(DType dtype) {
  if (dtype.IsPtr() || dtype.IsCollection()) {
    return "";
  }
  uint32_t base_type = dtype.GetFundamentalType();
  if (base_type >= DATA_VOID && base_type <= DATA_JSON) {
    return kFundamentalTypeStrs[base_type];
  }
  auto found = getDTypeNameMap().find(dtype.Control());
  if (found != getDTypeNameMap().end()) {
    return found->second;
  }
  return "";
}

DType DTypeFactory::GetDTypeByName(const std::string& name) {
  auto found = getNameDTypeMap().find(name);
  if (found != getNameDTypeMap().end()) {
    return found->second;
  }
  return {};
}

bool DTypeFactory::AddNameDType(const std::string& name, DType dtype) {
  bool r = getNameDTypeMap().emplace(name, dtype).second;
  if (r) {
    getDTypeNameMap().emplace(dtype.Control(), name);
  }
  return r;
}
DType DType::Key() const {
  DType result;
  if (container_type_ == COLLECTION_MAP || container_type_ == COLLECTION_UNORDERED_MAP) {
    result.t0_ = t0_;
  }
  return result;
}
DType DType::Elem() const {
  DType result;
  result.control_ = control_;
  if (result.container_type_ == COLLECTION_MAP || result.container_type_ == COLLECTION_UNORDERED_MAP) {
    result.t0_ = result.t1_;
  }
  result.container_type_ = 0;
  return result;
}

std::string DType::GetTypeString() const {
  auto base_dtype = Elem().PtrTo();
  auto base_name = DTypeFactory::GetNameByDType(base_dtype);
  if (base_name.empty()) {
    return "";
  }
  std::string name(base_name);
  if (IsCollection()) {
    std::string_view collection_type = kCollectionTypeStrs[container_type_];
    if (IsMap() || IsUnorderedMap() || IsTuple()) {
      auto key_dtype = Key();
      auto key_name = DTypeFactory::GetNameByDType(key_dtype);
      if (key_name.empty()) {
        return "";
      }
      name = fmt::format("{}<{},{}>", collection_type, key_name, name);
    } else {
      name = fmt::format("{}<{}>", collection_type, name);
    }
  }
  if (IsPtr()) {
    name.append(1, '*');
  }
  return name;
}

std::string DType::ToString() const {
  auto base_dtype = Elem().PtrTo();
  auto base_name = DTypeFactory::GetNameByDType(base_dtype);
  if (base_name.empty()) {
    return fmt::format("[type:{}, ptr:{}, container:{}]", GetFundamentalType(), ptr_bit_,
                       kCollectionTypeStrs[container_type_]);
  }
  std::string name(base_name);
  std::string_view collection_type;
  if (IsCollection()) {
    collection_type = kCollectionTypeStrs[container_type_];
    name = fmt::format("{}<{}>", collection_type, name);
  }
  if (IsPtr()) {
    name.append(1, '*');
  }
  return fmt::format("[{}]", name);
}

std::string DType::Demangle(const char* name) {
  int status = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (0 != status) {
    RUDF_ERROR("Failed to demangle {} with status:{}", name, status);
    free(demangled);
    return "unknown";
  }
  std::string ret = demangled;
  free(demangled);
  return ret;
}

bool DType::CanCastTo(DType other) const {
  if (*this == other) {
    return true;
  }
  if (IsVoid()) {
    return true;
  }
  if (IsNumber() && other.IsNumber()) {
    return true;
  }
  if (other.IsStringView()) {
    if (IsPtr()) {
      if (PtrTo().IsString() || PtrTo().IsFlatbuffersString()) {
        return true;
      }
    }
  }
  return false;
}

std::vector<DType> DType::ExtractTupleDtypes() const {
  // if (!IsTuple()) {
  //   return {};
  // }
  if (!IsTuple()) {
    return {*this};
  }
  std::vector<DType> dtypes;
  if (t0_ == 0) {
    return dtypes;
  }
  dtypes.emplace_back(DType(static_cast<FundamentalType>(t0_)));
  if (t1_ == 0) {
    return dtypes;
  }
  dtypes.emplace_back(DType(static_cast<FundamentalType>(t1_)));
  if (t2_ == 0) {
    return dtypes;
  }
  dtypes.emplace_back(DType(static_cast<FundamentalType>(t2_)));
  if (t3_ == 0) {
    return dtypes;
  }
  dtypes.emplace_back(DType(static_cast<FundamentalType>(t3_)));
  return dtypes;
}

uint32_t DType::TupleSize() const {
  if (!IsTuple()) {
    return 0;
  }
  if (t0_ == 0) {
    return 0;
  }
  if (t1_ == 0) {
    return 1;
  }
  if (t2_ == 0) {
    return 2;
  }
  if (t3_ == 0) {
    return 3;
  }
  return 4;
}

uint32_t DType::ByteSize() const {
  if (IsPtr()) {
    return 8;
  }
  if (IsAbslSpan()) {
    return 16;
  } else if (IsSimdVector()) {
    return 16;
  }
  switch (t0_) {
    case DATA_U8:
    case DATA_I8: {
      return 1;
    }
    case DATA_U16:
    case DATA_I16: {
      return 2;
    }
    case DATA_U32:
    case DATA_I32:
    case DATA_F32: {
      return 4;
    }
    case DATA_VOID:
    case DATA_F64:
    case DATA_U64:
    case DATA_I64: {
      return 8;
    }
    case DATA_STD_STRING_VIEW:
    case DATA_STRING_VIEW: {
      return 16;
    }
    default: {
      return 0;
    }
  }
}

template <typename T>
uint64_t cast_to(DType dtype, uint64_t val) {
  T result;
  switch (dtype.GetFundamentalType()) {
    case DATA_U8:
    case DATA_U16:
    case DATA_U32:
    case DATA_U64: {
      result = static_cast<T>(val);
      break;
    }
    case DATA_I8: {
      int8_t iv = static_cast<int8_t>(static_cast<uint8_t>(val));
      result = static_cast<T>(iv);
    }
    case DATA_I16: {
      int16_t iv = static_cast<int16_t>(static_cast<uint16_t>(val));
      result = static_cast<T>(iv);
      break;
    }
    case DATA_I32: {
      int32_t iv = static_cast<int32_t>(static_cast<uint32_t>(val));
      result = static_cast<T>(iv);
      break;
    }
    case DATA_I64: {
      result = static_cast<T>(static_cast<int64_t>(val));
      break;
    }
    case DATA_F32: {
      float dv;
      uint32_t uv = static_cast<uint32_t>(val);
      memcpy(&dv, &uv, sizeof(float));
      result = static_cast<T>(dv);
      break;
    }
    case DATA_F64: {
      double dv;
      memcpy(&dv, &val, sizeof(val));
      result = static_cast<T>(dv);
      break;
    }
    default: {
      abort();
    }
  }

  if constexpr (sizeof(T) == 8) {
    uint64_t bits;
    memcpy(&bits, &result, sizeof(T));
    return bits;
  } else if constexpr (sizeof(T) == 4) {
    uint32_t bits;
    memcpy(&bits, &result, sizeof(T));
    return bits;
  } else if constexpr (sizeof(T) == 2) {
    uint16_t bits;
    memcpy(&bits, &result, sizeof(T));
    return bits;
  } else if constexpr (sizeof(T) == 1) {
    uint8_t bits;
    memcpy(&bits, &result, sizeof(T));
    return bits;
  } else {
    abort();
  }
}

uint64_t convert_to(uint64_t val, DType src_dtype, DType dst_type) {
  if (src_dtype == dst_type) {
    return val;
  }

  if (!src_dtype.IsFundamental() || !dst_type.IsFundamental()) {
    abort();
    return val;
  }
  switch (dst_type.GetFundamentalType()) {
    case DATA_U8: {
      return cast_to<uint8_t>(src_dtype, val);
    }
    case DATA_U16: {
      return cast_to<uint16_t>(src_dtype, val);
    }
    case DATA_U32: {
      return cast_to<uint32_t>(src_dtype, val);
    }
    case DATA_U64: {
      return cast_to<uint64_t>(src_dtype, val);
    }
    case DATA_I8: {
      return cast_to<int8_t>(src_dtype, val);
    }
    case DATA_I16: {
      return cast_to<int16_t>(src_dtype, val);
    }
    case DATA_I32: {
      return cast_to<int32_t>(src_dtype, val);
    }
    case DATA_I64: {
      return cast_to<int64_t>(src_dtype, val);
    }
    case DATA_F64: {
      return cast_to<double>(src_dtype, val);
    }
    case DATA_F32: {
      return cast_to<float>(src_dtype, val);
    }
    default: {
      abort();
    }
  }
}

}  // namespace rapidudf