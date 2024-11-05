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
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "rapidudf/arena/arena.h"
#include "rapidudf/common/AtomicIntrusiveLinkedList.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/vector.h"

namespace rapidudf {

using CleanupFunc = std::function<void()>;
struct CleanupFuncWrapper {
  CleanupFunc func;
  CleanupFuncWrapper(CleanupFunc&& f) : func(std::move(f)) {}
  ::rapiduf::folly::AtomicIntrusiveLinkedListHook<CleanupFuncWrapper> _hook;
  using List = ::rapiduf::folly::AtomicIntrusiveLinkedList<CleanupFuncWrapper, &CleanupFuncWrapper::_hook>;
};
class Context {
 public:
  static constexpr uint32_t kByteLanes = 32;  // 256bit
  using CleanupFunc = std::function<void()>;
  Context(Arena* arena = nullptr);

  void Reset();

  uint8_t* ArenaAllocate(size_t n);

  template <typename T, typename... Args>
  T* New(Args&&... args) {
    auto tmp = std::make_unique<T>(std::forward<Args>(args)...);
    T* p = tmp.get();
    Own(std::move(tmp));
    return p;
  }

  template <typename T>
  simd::VectorData NewSimdVector(size_t lanes, size_t n, bool temporary = false) {
    using number_t = typename simd::InternalType<T>::internal_type;
    size_t element_size = get_arena_element_size(n, lanes);
    uint32_t byte_size = sizeof(number_t) * element_size;
    uint8_t* arena_data = ArenaAllocate(byte_size);
    simd::VectorData vec(arena_data, n, byte_size);
    if (temporary) {
      vec.SetTemporary(true);
    }
    vec.SetReadonly(false);
    return vec;
  }

  template <typename T>
  auto NewSimdVector(const std::vector<T*>& data) {
    uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(Pointer));
    Pointer* ptrs = reinterpret_cast<Pointer*>(arena_data);
    for (size_t i = 0; i < data.size(); i++) {
      *(ptrs + i) = Pointer(data[i]);
    }
    return simd::Vector<Pointer>(ptrs, data.size());
  }

  template <typename T>
  auto NewSimdVector(const std::vector<const T*>& data) {
    uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(Pointer));
    Pointer* ptrs = reinterpret_cast<Pointer*>(arena_data);
    for (size_t i = 0; i < data.size(); i++) {
      *(ptrs + i) = Pointer(data[i]);
    }
    return simd::Vector<Pointer>(ptrs, data.size());
  }

  template <typename T>
  auto NewSimdVector(const std::vector<T>& data) {
    if constexpr (std::is_same_v<bool, T>) {
      size_t byte_size = data.size() / 8;
      if (data.size() % 8 > 0) {
        byte_size++;
      }
      byte_size = get_arena_element_size(byte_size, 32);  // at least 32bytes
      uint8_t* arena_data = ArenaAllocate(byte_size);
      uint64_t* bits = reinterpret_cast<uint64_t*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        size_t bits_idx = i / 64;
        size_t bits_cursor = i % 64;
        if (data[i]) {
          bits[bits_idx] = bits64_set(bits[bits_idx], bits_cursor);
        } else {
          bits[bits_idx] = bits64_clear(bits[bits_idx], bits_cursor);
        }
      }
      simd::VectorData vdata(arena_data, data.size(), byte_size);
      vdata.SetReadonly(false);
      return simd::Vector<Bit>(vdata);
    } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<StringView, T> ||
                         is_std_array_v<T>) {
      return simd::Vector<T>(data);
    } else if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(StringView));
      StringView* strs = reinterpret_cast<StringView*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(strs + i) = StringView(data[i]);
      }
      return simd::Vector<StringView>(strs, data.size());
    } else {
      // static_assert(sizeof(T) == -1, "unsupported type to NewSimdVector");
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(Pointer));
      Pointer* ptrs = reinterpret_cast<Pointer*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(ptrs + i) = Pointer(&data[i]);
      }
      auto ret = simd::Vector<Pointer>(ptrs, data.size());
      ret.SetReadonly(false);
      return ret;
    }
  }

  template <typename T>
  auto NewSimdVector(std::vector<T>& data) {
    const std::vector<T>& const_ref = data;
    auto val = NewSimdVector(const_ref);
    val.GetVectorData().SetReadonly(false);
    return val;
  }

  template <typename T>
  void Own(std::unique_ptr<T>&& p) {
    auto* pp = p.release();
    auto f = [pp] { delete pp; };
    cleanups_.insertHead(new CleanupFuncWrapper(std::move(f)));
  }

  template <typename T, typename D>
  void Own(T* p, D d) {
    auto f = [p, d] { d(p); };
    cleanups_.insertHead(new CleanupFuncWrapper(std::move(f)));
  }

  void SetHasNan(bool v = true) { has_nan_ = v; }
  bool HasNan() const { return has_nan_; }

  ~Context();

 private:
  Arena& GetArena();

  std::unique_ptr<Arena> own_arena_;
  Arena* arena_ = nullptr;
  // absl::flat_hash_set<const uint8_t*> allocated_arena_ptrs_;
  // std::vector<CleanupFunc> cleanups_;
  CleanupFuncWrapper::List cleanups_;
  bool has_nan_ = false;
};
}  // namespace rapidudf