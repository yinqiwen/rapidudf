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
#include <memory>
#include <string>
#include "absl/types/span.h"
#include "rapidudf/context/context.h"
#include "rapidudf/memory/arena.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/reflect/struct.h"
#include "rapidudf/table/row.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {
namespace table {

struct ColumnField {
  std::string name;
  reflect::Field field;
  const RowSchema* schema = nullptr;
  uint32_t field_idx = 0;
  bool writ_back_updates = false;

  const ::google::protobuf::FieldDescriptor* GetProtobufField() const { return schema->pb_desc->field(field_idx); }
  const StructMember* GetStructField() const { return schema->struct_members->at(field_idx); }
};

class Column {
 public:
  Column(const ColumnField* field, VectorBase* vec)
      : field_(field), vector_(vec), memory_(nullptr), memory_size_(0), own_memory_(false) {}
  ~Column();

  const std::string& Name() const { return field_->name; }
  size_t GetFieldOffset() const { return field_->field.bytes_offset; }
  const DType& GetFieldDType() const { return field_->field.dtype; }
  const RowSchema* GetSchema() const { return field_->schema; }
  bool WriteBackUpdates() const { return field_->writ_back_updates; }

  const StructMember* GetStructField() const { return field_->GetStructField(); }
  const ::google::protobuf::FieldDescriptor* GetProtobufField() const { return field_->GetProtobufField(); }
  bool HasStructScheme() const { return field_->schema->struct_members != nullptr; }

  uint8_t* ReserveMemory(size_t element_size);
  void SetData(const VectorBase& vec, size_t memory_size, bool own);
  void Clear();
  void Unload();

  bool GetBool(const uint8_t* obj) const;
  uint8_t GetU8(const uint8_t* obj) const;
  uint16_t GetU16(const uint8_t* obj) const;
  uint32_t GetU32(const uint8_t* obj) const;
  uint64_t GetU64(const uint8_t* obj) const;
  int8_t GetI8(const uint8_t* obj) const;
  int16_t GetI16(const uint8_t* obj) const;
  int32_t GetI32(const uint8_t* obj) const;
  int64_t GetI64(const uint8_t* obj) const;
  float GetF32(const uint8_t* obj) const;
  double GetF64(const uint8_t* obj) const;
  StringView GetString(const uint8_t* obj) const;

  void GetRepeatedBool(const uint8_t* obj, absl::Span<const bool>* span) const;
  void GetRepeatedU8(const uint8_t* obj, absl::Span<const uint8_t>* span) const;
  void GetRepeatedU16(const uint8_t* obj, absl::Span<const uint16_t>* span) const;
  void GetRepeatedU32(const uint8_t* obj, absl::Span<const uint32_t>* span) const;
  void GetRepeatedU64(const uint8_t* obj, absl::Span<const uint64_t>* span) const;
  void GetRepeatedI8(const uint8_t* obj, absl::Span<const int8_t>* span) const;
  void GetRepeatedI16(const uint8_t* obj, absl::Span<const int16_t>* span) const;
  void GetRepeatedI32(const uint8_t* obj, absl::Span<const int32_t>* span) const;
  void GetRepeatedI64(const uint8_t* obj, absl::Span<const int64_t>* span) const;
  void GetRepeatedF32(const uint8_t* obj, absl::Span<const float>* span) const;
  void GetRepeatedF64(const uint8_t* obj, absl::Span<const double>* span) const;
  void GetRepeatedString(Context& ctx, const uint8_t* obj, absl::Span<const StringView>* span);

 private:
  template <typename T>
  void GetRepeatedElement(const uint8_t* obj, absl::Span<const T>* span) const;

  const ColumnField* field_ = nullptr;
  VectorBase* vector_ = nullptr;
  std::unique_ptr<Arena> arena_;
  uint8_t* memory_ = nullptr;
  size_t memory_size_ = 0;
  bool own_memory_ = false;
};

}  // namespace table
}  // namespace rapidudf