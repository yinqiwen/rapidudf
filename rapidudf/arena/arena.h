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
#include <vector>
#include "rapidudf/arena/leveldb_arena.h"
namespace rapidudf {
class Arena {
 public:
  using CleanupFunc = std::function<void()>;
  Arena() { arena_ = std::make_unique<leveldb::Arena>(); }
  uint8_t* Allocate(size_t n) {
    //  return google::protobuf::Arena::CreateArray<uint8_t>(&arena_, n);
    return reinterpret_cast<uint8_t*>(arena_->AllocateAligned(n));
  }
  void Reset() {
    arena_ = std::make_unique<leveldb::Arena>();
    for (auto& f : cleanups_) {
      f();
    }
    cleanups_.clear();
  }

  size_t MemoryUsage() const { return arena_->MemoryUsage(); }

  template <typename T>
  void Own(std::unique_ptr<T>&& p) {
    auto* pp = p.release();
    auto f = [pp] { delete pp; };
    cleanups_.emplace_back(std::move(f));
  }

 private:
  // google::protobuf::Arena arena_;
  std::unique_ptr<leveldb::Arena> arena_;
  std::vector<CleanupFunc> cleanups_;
};
}  // namespace rapidudf