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

#include <fmt/core.h>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "fmt/format.h"

#include "rapidudf/common/AtomicIntrusiveLinkedList.h"
#include "rapidudf/common/allign.h"
#include "rapidudf/memory/arena.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

namespace rapidudf {

using CleanupFunc = std::function<void()>;
struct CleanupFuncWrapper {
  CleanupFunc func;
  CleanupFuncWrapper(CleanupFunc&& f) : func(std::move(f)) {}
  ::rapiduf::folly::AtomicIntrusiveLinkedListHook<CleanupFuncWrapper> _hook;
  using List = ::rapiduf::folly::AtomicIntrusiveLinkedList<CleanupFuncWrapper, &CleanupFuncWrapper::_hook>;
};
template <typename T>
using PtrValidateFunc = std::function<bool(T*)>;

class Context {
 public:
  using CleanupFunc = std::function<void()>;

  Context(ThreadCachedArena* arena = nullptr);

  uint8_t* ArenaAllocate(size_t n);

  size_t ArenaMemoryUsage() const { return arena_->MemoryUsage(); }

  template <typename T, typename... Args>
  T* New(Args&&... args) {
    auto tmp = std::make_unique<T>(std::forward<Args>(args)...);
    return Own(std::move(tmp));
  }
  template <typename T, typename... Args>
  ArenaObjPtr<T> ArenaNew(Args&&... args) {
    return arena_->New<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  T* GetPtr(const PtrValidateFunc<T>&& validate, Args&&... args) {
    uint32_t tid = GetTypeId<T>();
    auto found = ptrs_.find(tid);
    if (found != ptrs_.end()) {
      T* p = reinterpret_cast<T*>(found->second);
      if (!validate || validate(p)) {
        return p;
      }
    }
    auto tmp = std::make_unique<T>(std::forward<Args>(args)...);
    T* p = Own(std::move(tmp));
    ptrs_[tid] = p;
    return p;
  }

  template <typename T>
  Vector<T> NewEmptyVector(size_t n, size_t capacity = 0) {
    uint32_t byte_size = 0;
    if constexpr (std::is_same_v<Bit, T> || std::is_same_v<bool, T>) {
      byte_size = n / 8;
      if (n % 8 > 0) {
        byte_size++;
      }
    } else {
      byte_size = sizeof(T) * n;
    }
    byte_size = align_to<uint32_t>(byte_size, 8);
    if (capacity < n) {
      capacity = n;
    }
    uint8_t* arena_data = ArenaAllocate(byte_size);
    return Vector<T>(reinterpret_cast<const T*>(arena_data), n, capacity, false);
  }

  template <typename T>
  auto NewVector(const std::vector<T>& data, const VectorBuildOptions& options = {}) {
    if constexpr (std::is_pointer_v<T>) {
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(Pointer));
      Pointer* ptrs = reinterpret_cast<Pointer*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(ptrs + i) = Pointer(data[i]);
      }
      return Vector<Pointer>(ptrs, data.size(), data.size(), false);
    } else if constexpr (std::is_same_v<T, bool>) {
      size_t byte_size = data.size() / 8;
      if (data.size() % 8 > 0) {
        byte_size++;
      }
      uint8_t* arena_data = ArenaAllocate(byte_size);
      uint8_t* bits = reinterpret_cast<uint8_t*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        size_t bits_idx = i / 8;
        size_t bits_cursor = i % 8;
        if (data[i]) {
          bits[bits_idx] = bit_set(bits[bits_idx], bits_cursor);
        } else {
          bits[bits_idx] = bit_clear(bits[bits_idx], bits_cursor);
        }
      }
      return Vector<Bit>(reinterpret_cast<Bit*>(bits), data.size(), data.size(), false);
    } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<StringView, T>) {
      if (options.clone) {
        uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(T));
        memcpy(arena_data, data.data(), data.size() * sizeof(T));
        return Vector<T>(reinterpret_cast<T*>(arena_data), data.size(), data.size(), false);
      } else {
        return Vector<T>(data.data(), data.size(), data.capacity(), options.readonly);
      }
    } else if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(StringView));
      StringView* strs = reinterpret_cast<StringView*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(strs + i) = StringView(data[i]);
      }
      return Vector<StringView>(strs, data.size(), data.size(), false);
    } else {
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(Pointer));
      Pointer* ptrs = reinterpret_cast<Pointer*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(ptrs + i) = Pointer(&data[i]);
      }
      return Vector<Pointer>(ptrs, data.size(), data.size(), false);
    }
  }

  template <typename... T>
  std::string_view NewString(fmt::format_string<T...> fmt, T&&... args) {
    size_t n = fmt::formatted_size(fmt, args...);
    if (n == 0) {
      return "";
    }
    char* data = reinterpret_cast<char*>(ArenaAllocate(n));
    fmt::format_to(data, fmt, args...);
    return std::string_view(data, n);
  }

  template <typename T>
  T* Own(std::unique_ptr<T>&& p) {
    auto* pp = p.release();
    auto f = [pp] { delete pp; };
    cleanups_.insertHead(new CleanupFuncWrapper(std::move(f)));
    return pp;
  }

  template <typename T>
  T* Own(std::shared_ptr<T>& p) {
    auto* pp = p.get();
    auto f = [p] { p.reset(); };
    cleanups_.insertHead(new CleanupFuncWrapper(std::move(f)));
    return pp;
  }

  template <typename T, typename D>
  void Own(T* p, D d) {
    auto f = [p, d] { d(p); };
    cleanups_.insertHead(new CleanupFuncWrapper(std::move(f)));
  }

  void SetHasNan(bool v = true) { has_nan_ = v; }
  bool HasNan() const { return has_nan_; }

  void Reset();

  ~Context();

 private:
  static uint32_t NextTypeId();

  template <typename T>
  static uint32_t GetTypeId() {
    static uint32_t id = NextTypeId();
    return id;
  }
  ThreadCachedArena& GetArena();

  std::unique_ptr<ThreadCachedArena> own_arena_;
  ThreadCachedArena* arena_ = nullptr;
  using PtrMap = absl::flat_hash_map<uint32_t, void*>;
  PtrMap ptrs_;

  CleanupFuncWrapper::List cleanups_;
  bool has_nan_ = false;
};
}  // namespace rapidudf