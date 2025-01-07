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

#include <gtest/gtest.h>
#include <string>
#include <type_traits>

#include "rapidudf/log/log.h"
#include "rapidudf/memory/arena.h"
#include "rapidudf/memory/arena_container.h"
#include "rapidudf/memory/arena_string.h"

using namespace rapidudf;
TEST(Arena, simple) {
  ThreadCachedArena arena;
  ThreadCachedArenaAllocator<char> alloc(arena);
  arena::String s(alloc);

  auto p = arena.New<arena::String>();
  p->assign("aaa");

  std::string_view v(*p);

  RUDF_INFO("###{} {}", std::is_trivially_destructible<arena::String>::value, v);

  auto pp = arena.New<::arena::Vector<::arena::String>>();
  pp->emplace_back(std::move(*p));
  RUDF_INFO("###{} {} ", pp->size(), std::is_trivially_destructible<std::string_view>::value);

  auto pmap = arena.New<::arena::HashMap<int, ::arena::String>>();
  RUDF_INFO("###{}  ", pmap->size());

  pmap->emplace(1, *p);

  RUDF_INFO("###{}  ", std::is_trivially_destructible<::arena::HashMap<int, ::arena::String>>::value);

  auto x = arena.New<int>();
  *x = 111;
  RUDF_INFO("###{}  ", *x);

  // ThreadCachedArena thread_safe_arena;
  // auto p1 = thread_safe_arena.New<arena::String>();
  // p1->assign("aaa");

  // RUDF_INFO("###cmp:{}", *p == *p1);
}