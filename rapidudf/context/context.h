/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
class Context {
 public:
  static constexpr uint32_t kByteLanes = 32;  // 256bit
  using CleanupFunc = std::function<void()>;
  Context(Arena* arena = nullptr);

  void Reset();

  uint8_t* ArenaAllocate(size_t n);

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
    return vec;
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
      return simd::Vector<Bit>(vdata);
    } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<StringView, T>) {
      return simd::Vector<T>(data);
    } else if constexpr (std::is_same_v<std::string, T> || std::is_same_v<std::string_view, T>) {
      uint8_t* arena_data = ArenaAllocate(data.size() * sizeof(StringView));
      StringView* strs = reinterpret_cast<StringView*>(arena_data);
      for (size_t i = 0; i < data.size(); i++) {
        *(strs + i) = StringView(data[i]);
      }
      return simd::Vector<StringView>(strs, data.size());
    } else {
      static_assert(sizeof(T) == -1, "unsupported type to NewSimdVector");
    }
  }

  template <typename T>
  void Own(std::unique_ptr<T>&& p) {
    auto* pp = p.release();
    auto f = [pp] { delete pp; };
    cleanups_.emplace_back(std::move(f));
  }

  template <typename T>
  bool IsTemporary(simd::Vector<T> data) {
    bool is_temporary = data.IsTemporary();
    if (!is_temporary) {
      return false;
    }
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.Data());
    return allocated_arena_ptrs_.count(ptr) > 0;
  }

  ~Context();

 private:
  Arena& GetArena();

  std::unique_ptr<Arena> own_arena_;
  Arena* arena_ = nullptr;
  absl::flat_hash_set<const uint8_t*> allocated_arena_ptrs_;
  std::vector<CleanupFunc> cleanups_;
};
}  // namespace rapidudf