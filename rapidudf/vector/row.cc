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
#include "rapidudf/vector/row.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/functions/simd/vector_misc.h"
namespace rapidudf {
namespace simd {
void Rows::Filter(Vector<Bit> bits) { pointers_ = functions::simd_vector_filter(ctx_, pointers_, bits); }

void Rows::Truncate(size_t k) {
  if (pointers_.Size() > k) {
    pointers_.Resize(k);
  }
}
void Rows::Truncate(size_t pos, size_t k) {
  if (pos >= pointers_.Size()) {
    return;
  }
  if ((pos + k) > pointers_.Size()) {
    k = pointers_.Size() - pos;
  }
  pointers_ = pointers_.SubVector(pos, k);
}

void Rows::Gather(Vector<int32_t> indices) { pointers_ = functions::simd_vector_gather(ctx_, pointers_, indices); }
void Rows::Append(const std::vector<const uint8_t*>& objs) {
  objs_.insert(objs_.end(), objs.begin(), objs.end());
  pointers_ = ctx_.NewSimdVector(objs_);
}
absl::Status Rows::Insert(size_t pos, const uint8_t* ptr) {
  if (pos > objs_.size()) {
    return absl::OutOfRangeError("too large pos to insert");
  }
  objs_.insert(objs_.begin() + pos, ptr);
  return absl::OkStatus();
}
}  // namespace simd
}  // namespace rapidudf