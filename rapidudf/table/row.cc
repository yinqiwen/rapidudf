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
#include "rapidudf/table/row.h"
#include <cstring>
#include <vector>
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/types/pointer.h"

namespace rapidudf {
namespace table {

void Rows::SetPointers(Vector<Pointer>* new_pointers) {
  // if (objs_.size() < new_pointers->Size()) {
  //   objs_.resize(new_pointers->Size());
  // }
  // memcpy(&objs_[0], new_pointers->Data(), new_pointers->Size() * sizeof(const uint8_t*));
  objs_.resize(new_pointers->Size());
  obj_ptrs_ = new_pointers;
}
void Rows::Reset(std::vector<const uint8_t*>&& objs) {
  objs_ = std::move(objs);
  dirty_ = true;
}
void Rows::Filter(Vector<Bit>* bits) {
  auto new_pointers = functions::simd_vector_filter(ctx_, GetRowPtrs(), bits);
  SetPointers(new_pointers);
}

void Rows::Truncate(size_t k) {
  if (objs_.size() > k) {
    objs_.resize(k);
    dirty_ = true;
  }
}
void Rows::Truncate(size_t pos, size_t k) {
  if (pos >= objs_.size()) {
    return;
  }
  if ((pos + k) > objs_.size()) {
    k = objs_.size() - pos;
  }
  objs_ = std::vector<const uint8_t*>(objs_.begin() + pos, objs_.begin() + pos + k);
  dirty_ = true;
}

void Rows::Gather(Vector<int32_t>* indices) {
  auto new_pointers = functions::simd_vector_gather(ctx_, GetRowPtrs(), indices);
  SetPointers(new_pointers);
}
void Rows::Append(const std::vector<const uint8_t*>& objs) { objs_.insert(objs_.end(), objs.begin(), objs.end()); }
absl::Status Rows::Insert(size_t pos, const uint8_t* ptr) {
  if (pos > objs_.size()) {
    return absl::OutOfRangeError("too large pos to insert");
  }
  objs_.insert(objs_.begin() + pos, ptr);
  return absl::OkStatus();
}
Vector<Pointer>* Rows::GetRowPtrs() {
  if (obj_ptrs_ == nullptr || dirty_) {
    absl::Span<Pointer> pointer_view = Pointer::Wrap(objs_);
    obj_ptrs_ = ctx_.NewVector(pointer_view);
    dirty_ = false;
  }
  return obj_ptrs_;
}
}  // namespace table

}  // namespace rapidudf