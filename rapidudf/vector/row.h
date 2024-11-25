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
#include <cstdint>
#include <memory>
#include <vector>
#include "flatbuffers/flatbuffers.h"
#include "google/protobuf/descriptor.h"
#include "rapidudf/context/context.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/reflect/reflect.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/vector/vector.h"

namespace rapidudf {
namespace simd {

struct RowSchema {
  const ::google::protobuf::Descriptor* pb_desc = nullptr;
  const flatbuffers::TypeTable* fbs_table = nullptr;
  DType struct_dtype;
  const std::vector<const StructMember*>* struct_members = nullptr;

  explicit RowSchema(const ::google::protobuf::Descriptor* desc) { pb_desc = desc; }
  explicit RowSchema(const flatbuffers::TypeTable* fbs) { fbs_table = fbs; }
  explicit RowSchema(const DType& dtype) {
    struct_dtype = dtype;
    struct_members = Reflect::GetStructMembers(dtype);
  }

  bool operator==(const RowSchema& s) const {
    return pb_desc == s.pb_desc && fbs_table == s.fbs_table && struct_dtype == s.struct_dtype;
  }
};
using RowSchemaPtr = std::unique_ptr<RowSchema>;

class Rows {
 public:
  explicit Rows(Context& ctx, std::vector<const uint8_t*>&& objs, const RowSchema& s)
      : ctx_(ctx), objs_(std::move(objs)), schema_(s) {
    pointers_ = ctx.NewSimdVector(objs_);
  }
  absl::Status Insert(size_t pos, const uint8_t* ptr);
  void Append(const std::vector<const uint8_t*>& objs);

  const RowSchema& GetSchema() const { return schema_; }
  const Vector<Pointer>& GetRowPtrs() const { return pointers_; }

  size_t RowCount() const { return pointers_.Size(); }

  void Filter(Vector<Bit> bits);
  void Truncate(size_t k);
  void Truncate(size_t pos, size_t k);
  void Gather(Vector<int32_t> indices);

 private:
  Context& ctx_;
  std::vector<const uint8_t*> objs_;
  Vector<Pointer> pointers_;
  RowSchema schema_;
};
using RowsPtr = std::unique_ptr<Rows>;
}  // namespace simd
}  // namespace rapidudf