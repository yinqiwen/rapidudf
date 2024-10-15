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

#pragma once
#include <functional>
#include <vector>
#include "google/protobuf/arena.h"
namespace rapidudf {
class Arena {
 public:
  using CleanupFunc = std::function<void()>;
  uint8_t* Allocate(size_t n) { return google::protobuf::Arena::CreateArray<uint8_t>(&arena_, n); }
  void Reset() {
    arena_.Reset();
    for (auto& f : cleanups_) {
      f();
    }
    cleanups_.clear();
  }

  template <typename T>
  void Own(std::unique_ptr<T>&& p) {
    auto* pp = p.release();
    auto f = [pp] { delete pp; };
    cleanups_.emplace_back(std::move(f));
  }

 private:
  google::protobuf::Arena arena_;
  std::vector<CleanupFunc> cleanups_;
};
}  // namespace rapidudf