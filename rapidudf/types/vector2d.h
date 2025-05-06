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
#include <vector>
#include "absl/types/span.h"

namespace rapidudf {
template <typename T>
class JaggedVectors {
 public:
  void Add(const std::vector<T>& cotent) {
    offsets_.push_back(content_.size());
    content_.insert(content_.end(), cotent.begin(), cotent.end());
  }
  absl::Span<T> Get(size_t index) const {
    return absl::MakeConstSpan(content_).subspan(GetOffset(index), GetLength(index));
  }
  size_t Size() const { return offsets_.size(); }

 private:
  size_t GetLength(size_t index) const { return offsets_[index] - GetOffset(index); }
  size_t GetOffset(size_t index) const {
    if (index == 0) {
      return 0;
    } else {
      return offsets_[index - 1];
    }
  }
  std::vector<uint32_t> offsets_;
  std::vector<T> content_;
};

template <typename T>
class FixedSizeVectors {
 public:
  explicit FixedSizeVectors(size_t fixed_size) : fixed_size_(fixed_size) {}
  bool Add(const std::vector<T>& cotent) {
    if (cotent.size() != fixed_size_) {
      return false;
    }
    content_.insert(content_.end(), cotent.begin(), cotent.end());
    return true;
  }
  absl::Span<T> Get(size_t index) const {
    return absl::MakeConstSpan(content_).subspan(index * fixed_size_, fixed_size_);
  }
  size_t Size() const { return content_.size() / fixed_size_; }

 private:
  std::vector<T> content_;
  size_t fixed_size_;
};
}  // namespace rapidudf