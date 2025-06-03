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
#include "rapidudf/table/column.h"
#include <array>
#include <string_view>
#include <type_traits>
#include <vector>
#include "flatbuffers/minireflect.h"
#include "google/protobuf/message.h"
#include "google/protobuf/reflection.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
namespace table {

Column::~Column() { Clear(); }

uint8_t* Column::ReserveMemory(size_t element_size) {
  DType element_dtype = GetFieldDType().Elem();
  size_t bytes_n = 0;
  if (element_dtype.IsBit()) {
    uint32_t n = element_size / 8;
    bytes_n = element_size % 8 > 0 ? n + 1 : n;
  } else if (element_dtype.IsArray() || element_dtype.IsVector()) {
    bytes_n = element_size * sizeof(absl::Span<uint8_t>);
  } else {
    bytes_n = element_dtype.ByteSize() * element_size;
  }
  if (memory_size_ < bytes_n) {
    Clear();
    memory_ = new uint8_t[bytes_n];
    memory_size_ = bytes_n;
    VectorBase alloc(memory_, 0, element_size, false);
    *vector_ = alloc;
    own_memory_ = true;
  }
  return memory_;
}

void Column::SetData(const VectorBase& vec, size_t memory_size, bool own) {
  Clear();
  own_memory_ = own;
  memory_ = const_cast<uint8_t*>(vec.Data<uint8_t>());
  // memory_ = memory;
  memory_size_ = memory_size;
  // VectorBase alloc(memory_, element_count, element_count, false);
  *vector_ = vec;
}

void Column::Clear() {
  if (own_memory_) {
    delete[] memory_;
  }
  memory_size_ = 0;
  memory_ = nullptr;
  VectorBase alloc(memory_, 0, 0, true);
  *vector_ = alloc;
}
void Column::Unload() { vector_->SetSize(0); }

bool Column::GetBool(const uint8_t* obj) const {
  if (obj == nullptr) {
    return false;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *ptr;
    } else {
      using func_t = bool (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetBool(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<bool>(ptr);
  }
  return false;
}
uint8_t Column::GetU8(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *ptr;
    } else {
      using func_t = uint8_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    return 0;
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<uint8_t>(ptr);
  }
  return 0;
}
uint16_t Column::GetU16(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const uint16_t*>(ptr);
    } else {
      using func_t = uint16_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    return 0;
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<uint16_t>(ptr);
  }
  return 0;
}
uint32_t Column::GetU32(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const uint32_t*>(ptr);
    } else {
      using func_t = uint32_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetUInt32(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<uint32_t>(ptr);
  }
  return 0;
}
uint64_t Column::GetU64(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const uint64_t*>(ptr);
    } else {
      using func_t = uint64_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetUInt64(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<uint64_t>(ptr);
  }
  return 0;
}
int8_t Column::GetI8(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const int8_t*>(ptr);
    } else {
      using func_t = int8_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    return 0;
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<int8_t>(ptr);
  }
  return 0;
}
int16_t Column::GetI16(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const int16_t*>(ptr);
    } else {
      using func_t = int16_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    return 0;
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<int16_t>(ptr);
  }
  return 0;
}
int32_t Column::GetI32(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const int32_t*>(ptr);
    } else {
      using func_t = int32_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetInt32(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<int32_t>(ptr);
  }
  return 0;
}
int64_t Column::GetI64(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const int64_t*>(ptr);
    } else {
      using func_t = int64_t (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetInt64(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<int64_t>(ptr);
  }
  return 0;
}

StringView Column::GetString(const uint8_t* obj) const {
  if (obj == nullptr) {
    return "";
  }
  if (field_->schema->struct_members != nullptr) {
    DType field_dtype;
    if (GetStructField()->HasField()) {
      field_dtype = *(GetStructField()->member_field_dtype);
    } else {
      field_dtype = (GetStructField()->member_func->return_type);
    }
    if (GetStructField()->HasField()) {
      if (field_dtype.IsString()) {
        const std::string* ptr = reinterpret_cast<const std::string*>(obj + GetStructField()->member_field_offset);
        return StringView(*ptr);
      } else if (field_dtype.IsStdStringView()) {
        const std::string_view* ptr =
            reinterpret_cast<const std::string_view*>(obj + GetStructField()->member_field_offset);
        return StringView(*ptr);
      } else {
        return "";
      }
    } else {
      if (field_dtype.IsStringPtr()) {
        using func_t = const std::string& (*)(void*);
        func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
        return StringView(f(const_cast<uint8_t*>(obj)));

      } else if (field_dtype.IsStdStringView()) {
        using func_t = std::string_view (*)(void*);
        func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
        return StringView(f(const_cast<uint8_t*>(obj)));
      } else {
        return "";
      }
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    static std::string empty;
    const std::string& ref = reflect->GetStringReference(*msg, GetProtobufField(), &empty);
    if (!ref.empty()) {
      return StringView(ref);
    } else {
      return "";
    }
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    ptr += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(ptr);
    const flatbuffers::String* str = reinterpret_cast<const flatbuffers::String*>(ptr);
    return StringView(reinterpret_cast<const char*>(str->Data()), str->size());
  }
  return 0;
}

float Column::GetF32(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const float*>(ptr);
    } else {
      using func_t = float (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetFloat(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<float>(ptr);
  }
  return 0;
}

double Column::GetF64(const uint8_t* obj) const {
  if (obj == nullptr) {
    return 0;
  }
  if (field_->schema->struct_members != nullptr) {
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      return *reinterpret_cast<const double*>(ptr);
    } else {
      using func_t = double (*)(void*);
      func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
      return f(const_cast<uint8_t*>(obj));
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    return reflect->GetDouble(*msg, GetProtobufField());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    return flatbuffers::ReadScalar<double>(ptr);
  }
  return 0;
}

template <typename T>
void Column::GetRepeatedElement(const uint8_t* obj, absl::Span<const T>* span) const {
  *span = {};
  if (obj == nullptr) {
    return;
  }
  if (field_->schema->struct_members != nullptr) {
    DType field_dtype;
    if (GetStructField()->HasField()) {
      field_dtype = *(GetStructField()->member_field_dtype);
    } else {
      field_dtype = (GetStructField()->member_func->return_type);
    }
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      if (field_dtype.IsArray()) {
        const T* data_ptr = reinterpret_cast<const T*>(ptr);
        size_t array_num = field_dtype.FixedArraySize();
        *span = absl::MakeConstSpan(data_ptr, array_num);
      } else if (field_dtype.IsVector()) {
        if constexpr (!std::is_same_v<bool, T>) {
          const std::vector<T>* data_ptr = reinterpret_cast<const std::vector<T>*>(ptr);
          *span = absl::MakeConstSpan(data_ptr->data(), data_ptr->size());
        }
      } else {
        return;
      }
    } else {
      if (field_dtype.IsPtr() && field_dtype.PtrTo().IsArray()) {
        using func_t = const T* (*)(void*);
        func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
        const T* data_ptr = f(const_cast<uint8_t*>(obj));
        size_t array_num = field_dtype.FixedArraySize();
        *span = absl::MakeConstSpan(data_ptr, array_num);
      } else if (field_dtype.IsPtr() && field_dtype.PtrTo().IsVector()) {
        if constexpr (!std::is_same_v<bool, T>) {
          using func_t = const std::vector<T>& (*)(void*);
          func_t f = reinterpret_cast<func_t>(GetStructField()->member_func->func);
          const std::vector<T>& data_vec = f(const_cast<uint8_t*>(obj));
          *span = absl::MakeConstSpan(data_vec.data(), data_vec.size());
        }
      } else {
        return;
      }
    }
  } else if (field_->schema->pb_desc != nullptr) {
    if constexpr (std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<uint64_t, T> ||
                  std::is_same_v<uint32_t, T> || std::is_same_v<int64_t, T> || std::is_same_v<int32_t, T> ||
                  std::is_same_v<bool, T>) {
      const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
      const ::google::protobuf::Reflection* reflect = msg->GetReflection();
      auto ret = reflect->GetRepeatedField<T>(*msg, GetProtobufField());
      *span = absl::MakeConstSpan(ret.data(), ret.size());
    }
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    ptr += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(ptr);
    auto vec = reinterpret_cast<const flatbuffers::Vector<T>*>(ptr);
    *span = absl::MakeConstSpan(vec->data(), vec->size());
  }
  return;
}

void Column::GetRepeatedBool(const uint8_t* obj, absl::Span<const bool>* span) const {
  GetRepeatedElement<bool>(obj, span);
}
void Column::GetRepeatedU8(const uint8_t* obj, absl::Span<const uint8_t>* span) const {
  GetRepeatedElement<uint8_t>(obj, span);
}
void Column::GetRepeatedU16(const uint8_t* obj, absl::Span<const uint16_t>* span) const {
  GetRepeatedElement<uint16_t>(obj, span);
}
void Column::GetRepeatedU32(const uint8_t* obj, absl::Span<const uint32_t>* span) const {
  GetRepeatedElement<uint32_t>(obj, span);
}
void Column::GetRepeatedU64(const uint8_t* obj, absl::Span<const uint64_t>* span) const {
  return GetRepeatedElement<uint64_t>(obj, span);
}
void Column::GetRepeatedI8(const uint8_t* obj, absl::Span<const int8_t>* span) const {
  GetRepeatedElement<int8_t>(obj, span);
}
void Column::GetRepeatedI16(const uint8_t* obj, absl::Span<const int16_t>* span) const {
  GetRepeatedElement<int16_t>(obj, span);
}
void Column::GetRepeatedI32(const uint8_t* obj, absl::Span<const int32_t>* span) const {
  GetRepeatedElement<int32_t>(obj, span);
}
void Column::GetRepeatedI64(const uint8_t* obj, absl::Span<const int64_t>* span) const {
  GetRepeatedElement<int64_t>(obj, span);
}

void Column::GetRepeatedF32(const uint8_t* obj, absl::Span<const float>* span) const {
  GetRepeatedElement<float>(obj, span);
}

void Column::GetRepeatedF64(const uint8_t* obj, absl::Span<const double>* span) const {
  GetRepeatedElement<double>(obj, span);
}

void Column::GetRepeatedString(Context& ctx, const uint8_t* obj, absl::Span<const StringView>* span) {
  *span = {};
  if (obj == nullptr) {
    return;
  }
  if (field_->schema->struct_members != nullptr) {
    DType field_dtype;
    if (GetStructField()->HasField()) {
      field_dtype = *(GetStructField()->member_field_dtype);
    } else {
      field_dtype = (GetStructField()->member_func->return_type);
    }
    if (GetStructField()->HasField()) {
      const uint8_t* ptr = obj + GetStructField()->member_field_offset;
      if (field_dtype.IsArray()) {
        size_t array_num = field_dtype.FixedArraySize();
        StringView* views = reinterpret_cast<StringView*>(ctx.ArenaAllocate(array_num * sizeof(StringView)));
        if (field_dtype.Elem().IsStdStringView()) {
          const std::string_view* data_ptr = reinterpret_cast<const std::string_view*>(ptr);
          for (size_t i = 0; i < array_num; i++) {
            views[i] = StringView(data_ptr[i]);
          }
          *span = absl::MakeConstSpan(views, array_num);
        } else if (field_dtype.Elem().IsString()) {
          const std::string* data_ptr = reinterpret_cast<const std::string*>(ptr);
          for (size_t i = 0; i < array_num; i++) {
            views[i] = StringView(data_ptr[i]);
          }
          *span = absl::MakeConstSpan(views, array_num);
        }
      } else if (field_dtype.IsVector()) {
        if (field_dtype.Elem().IsStdStringView()) {
          const std::vector<std::string_view>* data_ptr = reinterpret_cast<const std::vector<std::string_view>*>(ptr);
          if (data_ptr->empty()) {
            return;
          }
          StringView* views = reinterpret_cast<StringView*>(ctx.ArenaAllocate(data_ptr->size() * sizeof(StringView)));
          for (size_t i = 0; i < data_ptr->size(); i++) {
            views[i] = StringView((*data_ptr)[i]);
          }
          *span = absl::MakeConstSpan(views, data_ptr->size());
        } else if (field_dtype.Elem().IsString()) {
          const std::vector<std::string>* data_ptr = reinterpret_cast<const std::vector<std::string>*>(ptr);
          if (data_ptr->empty()) {
            return;
          }
          StringView* views = reinterpret_cast<StringView*>(ctx.ArenaAllocate(data_ptr->size() * sizeof(StringView)));
          for (size_t i = 0; i < data_ptr->size(); i++) {
            views[i] = StringView((*data_ptr)[i]);
          }
          *span = absl::MakeConstSpan(views, data_ptr->size());
        }
      }
    } else {
      if (field_dtype.IsPtr() && field_dtype.PtrTo().IsArray()) {
      } else if (field_dtype.IsPtr() && field_dtype.PtrTo().IsVector()) {
      } else {
        return;
      }
    }
  } else if (field_->schema->pb_desc != nullptr) {
    const ::google::protobuf::Message* msg = reinterpret_cast<const ::google::protobuf::Message*>(obj);
    const ::google::protobuf::Reflection* reflect = msg->GetReflection();
    auto ret = reflect->GetRepeatedPtrField<std::string>(*msg, GetProtobufField());
    StringView* views = reinterpret_cast<StringView*>(ctx.ArenaAllocate(ret.size() * sizeof(StringView)));
    for (size_t i = 0; i < ret.size(); i++) {
      views[i] = StringView(ret.Get(i));
    }
    *span = absl::MakeConstSpan(views, ret.size());
  } else if (field_->schema->fbs_table != nullptr) {
    const flatbuffers::Table* fbs = reinterpret_cast<const flatbuffers::Table*>(obj);
    const uint8_t* ptr =
        fbs->GetAddressOf(flatbuffers::FieldIndexToOffset(static_cast<flatbuffers::voffset_t>(field_->field_idx)));
    ptr += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(ptr);
    auto vec = reinterpret_cast<const flatbuffers::Vector<uint8_t>*>(ptr);
    auto elem_ptr = vec->Data();
    auto size = vec->size();
    StringView* views = reinterpret_cast<StringView*>(ctx.ArenaAllocate(size * sizeof(StringView)));
    for (size_t i = 0; i < size; i++) {
      auto val = elem_ptr;
      val += flatbuffers::ReadScalar<flatbuffers::uoffset_t>(val);
      auto str = reinterpret_cast<const flatbuffers::String*>(val);
      views[i] = StringView(reinterpret_cast<const char*>(str->Data()), str->size());
      elem_ptr += flatbuffers::InlineSize(flatbuffers::ET_STRING, field_->schema->fbs_table);
    }
    *span = absl::MakeConstSpan(views, size);
  }
  return;
}

}  // namespace table
}  // namespace rapidudf