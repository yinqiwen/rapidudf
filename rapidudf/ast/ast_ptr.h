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

#include <cstddef>

namespace rapidudf {
namespace ast {

// Lightweight copyable smart pointer for arena-allocated AST nodes.
// The arena (AstPool) owns the object's lifetime; AstPtr is a borrowed reference.
// Copyable so that std::variant containing AstPtr remains copy-constructible.
template <typename T>
class AstPtr {
  T* ptr_ = nullptr;

 public:
  AstPtr() = default;
  AstPtr(std::nullptr_t) : ptr_(nullptr) {}  // NOLINT
  explicit AstPtr(T* p) : ptr_(p) {}

  AstPtr(const AstPtr&) = default;
  AstPtr& operator=(const AstPtr&) = default;
  AstPtr(AstPtr&&) = default;
  AstPtr& operator=(AstPtr&&) = default;

  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  T* get() const { return ptr_; }

  explicit operator bool() const { return ptr_ != nullptr; }
  bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return ptr_ != nullptr; }
  bool operator==(const AstPtr& o) const { return ptr_ == o.ptr_; }
  bool operator!=(const AstPtr& o) const { return ptr_ != o.ptr_; }
};

}  // namespace ast
}  // namespace rapidudf
