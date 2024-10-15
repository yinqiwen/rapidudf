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
#include "rapidudf/context/context.h"
#include <memory>

namespace rapidudf {
Context::Context(Arena* arena) : arena_(arena) {
  if (nullptr == arena_) {
    own_arena_ = std::make_unique<Arena>();
    arena_ = own_arena_.get();
  }
}
Context::~Context() { Reset(); }
Arena& Context::GetArena() { return *arena_; }
uint8_t* Context::ArenaAllocate(size_t n) {
  uint8_t* p = GetArena().Allocate(n);
  // allocated_arena_ptrs_.insert(p);
  return p;
}

void Context::Reset() {
  GetArena().Reset();
  // allocated_arena_ptrs_.clear();
  // for (auto clean : cleanups_) {
  //   clean();
  // }
  cleanups_.sweep([](CleanupFuncWrapper* wrapper) {
    wrapper->func();
    delete wrapper;
  });
}
}  // namespace rapidudf