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
#include <type_traits>
#include <vector>

#include "rapidudf/memory/arena.h"

namespace rapidudf {
namespace ast {

// Arena-based memory pool for AST nodes.
// Wraps Arena (which uses folly::Arena internally) for bump allocation.
// Supports non-trivially-destructible types via destructor callbacks.
// Provides GetArena() for constructing arena-backed containers (AstVector, etc.).

// Allocator adapter for AST arena-backed containers.
// FallbackToStdAlloc=true allows default construction (for variants/locals).
template <typename T>
using AstAlloc = CxxAllocatorAdaptor<T, Arena, /*FallbackToStdAlloc=*/true>;

// Arena-backed vector for use in arena-allocated AST nodes.
template <typename T>
using AstVector = std::vector<T, AstAlloc<T>>;

class AstPool {
 public:
  AstPool() = default;

  // Allocate raw memory from the arena (bump allocation, amortized zero-cost).
  void* Allocate(size_t size) { return arena_.allocate(size); }

  // Access the underlying Arena for constructing arena-backed containers.
  Arena& GetArena() { return arena_; }

  // Allocate + construct an object.
  // If T has an allocator-accepting constructor, passes AstAlloc<T>.
  // Registers destructor callback if T is non-trivially-destructible.
  template <typename T, typename... Args>
  T* New(Args&&... args) {
    void* mem = Allocate(sizeof(T));
    T* obj;
    if constexpr (has_allocator_constructor_v<T, AstAlloc<T>, Args...>) {
      AstAlloc<T> alloc(arena_);
      obj = new (mem) T(std::forward<Args>(args)..., alloc);
    } else {
      obj = new (mem) T(std::forward<Args>(args)...);
    }
    if constexpr (!std::is_trivially_destructible_v<T>) {
      destructors_.push_back({obj, &Destroy<T>});
    }
    return obj;
  }

  // Run all registered destructors in reverse order, then reset the arena.
  // Arena memory blocks are retained for reuse (no malloc on next allocation round).
  void Reset() {
    for (auto it = destructors_.rbegin(); it != destructors_.rend(); ++it) {
      it->fn(it->ptr);
    }
    destructors_.clear();
    arena_.Reset();
  }

  ~AstPool() {
    for (auto it = destructors_.rbegin(); it != destructors_.rend(); ++it) {
      it->fn(it->ptr);
    }
    destructors_.clear();
  }

 private:
  using DestructFn = void (*)(void*);

  struct DestructorEntry {
    void* ptr;
    DestructFn fn;
  };

  template <typename T>
  static void Destroy(void* p) {
    static_cast<T*>(p)->~T();
  }

  Arena arena_;
  std::vector<DestructorEntry> destructors_;
};

}  // namespace ast
}  // namespace rapidudf
