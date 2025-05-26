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
#include <cstdint>
#include <vector>
#include "absl/types/span.h"

namespace rapidudf {
class Pointer {
 public:
  template <typename T>
  Pointer(const T* p) {
    ptr_val_ = reinterpret_cast<uintptr_t>(p);
  }
  Pointer(uint64_t v) { ptr_val_ = v; }
  template <typename T>
  T* As() const {
    return reinterpret_cast<T*>(ptr_val_);
  }

  bool IsNull() const { return ptr_val_ == 0; }

  template <typename R>
  static absl::Span<Pointer> Wrap(const std::vector<R*>& ptrs) {
    auto* ptr_data = ptrs.data();
    Pointer* p = const_cast<Pointer*>(reinterpret_cast<const Pointer*>(ptr_data));
    return absl::MakeSpan(p, ptrs.size());
  }

 private:
  uintptr_t ptr_val_;
};

}  // namespace rapidudf