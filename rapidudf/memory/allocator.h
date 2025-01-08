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

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

namespace rapidudf {
/**
 * CxxAllocatorAdaptor(copy from folly)
 *
 * A type conforming to C++ concept Allocator, delegating operations to an
 * unowned Inner which has this required interface:
 *
 *   void* allocate(std::size_t)
 *   void deallocate(void*, std::size_t)
 *
 * Note that Inner is *not* a C++ Allocator.
 */
template <typename T, class Inner, bool FallbackToStdAlloc = false>
class CxxAllocatorAdaptor : private std::allocator<T> {
 private:
  using Self = CxxAllocatorAdaptor<T, Inner, FallbackToStdAlloc>;

  template <typename U, typename UInner, bool UFallback>
  friend class CxxAllocatorAdaptor;

  Inner* inner_ = nullptr;

 public:
  using value_type = T;

  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

  template <bool X = FallbackToStdAlloc, std::enable_if_t<X, int> = 0>
  constexpr explicit CxxAllocatorAdaptor() {}

  constexpr explicit CxxAllocatorAdaptor(Inner& ref) : inner_(&ref) {}

  constexpr CxxAllocatorAdaptor(CxxAllocatorAdaptor const&) = default;

  template <typename U, std::enable_if_t<!std::is_same<U, T>::value, int> = 0>
  constexpr CxxAllocatorAdaptor(CxxAllocatorAdaptor<U, Inner, FallbackToStdAlloc> const& other)
      : inner_(other.inner_) {}

  CxxAllocatorAdaptor& operator=(CxxAllocatorAdaptor const& other) = default;

  template <typename U, std::enable_if_t<!std::is_same<U, T>::value, int> = 0>
  CxxAllocatorAdaptor& operator=(CxxAllocatorAdaptor<U, Inner, FallbackToStdAlloc> const& other) noexcept {
    inner_ = other.inner_;
    return *this;
  }

  T* allocate(std::size_t n) {
    if (FallbackToStdAlloc && inner_ == nullptr) {
      return std::allocator<T>::allocate(n);
    }
    return static_cast<T*>(inner_->allocate(sizeof(T) * n));
  }

  void deallocate(T* p, std::size_t n) {
    if (inner_ != nullptr) {
      inner_->deallocate(p, sizeof(T) * n);
    } else {
      assert(FallbackToStdAlloc);
      std::allocator<T>::deallocate(p, n);
    }
  }

  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new ((void*)p) U(std::forward<Args>(args)...);
  }

  template <typename U>
  void destroy(U* p) {
    if (inner_ != nullptr) {
      inner_->destroy(p);
    }
  }

  friend bool operator==(Self const& a, Self const& b) noexcept { return a.inner_ == b.inner_; }
  friend bool operator!=(Self const& a, Self const& b) noexcept { return a.inner_ != b.inner_; }

  template <typename U>
  struct rebind {
    using other = CxxAllocatorAdaptor<U, Inner, FallbackToStdAlloc>;
  };
};

/**
 * AllocatorHasTrivialDeallocate
 *
 * Unambiguously inherits std::integral_constant<bool, V> for some bool V.
 *
 * Describes whether a C++ Aallocator has trivial, i.e. no-op, deallocate().
 *
 * Also may be used to describe types which may be used with
 * CxxAllocatorAdaptor.
 */
template <typename Alloc>
struct AllocatorHasTrivialDeallocate : std::false_type {};

template <typename T, class Alloc>
struct AllocatorHasTrivialDeallocate<CxxAllocatorAdaptor<T, Alloc>> : AllocatorHasTrivialDeallocate<Alloc> {};

}  // namespace rapidudf