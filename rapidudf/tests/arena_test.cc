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
#include <vector>

#include "rapidudf/log/log.h"
#include "rapidudf/memory/arena.h"
#include "rapidudf/memory/arena_container.h"
#include "rapidudf/memory/arena_string.h"
#include "rapidudf/meta/type_traits.h"

using namespace rapidudf;

struct Test1 {
  int a;
  int b;
  std::string_view s;
};
struct Test2 {
  static constexpr bool destructor_disabled = true;
  int a;
  int b;
  ::arena::String s;
  Test2(const ThreadSafeArenaAllocator<char>& a) : s(a) {}
};

TEST(Arena, simple) {
  ThreadCachedArena arena;
  auto alloc = arena.GetAllocator();
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
  ASSERT_TRUE(std::is_trivially_destructible<Test1>::value);
  ASSERT_TRUE(is_destructor_disabled_v<Test2>);
  ASSERT_TRUE(is_destructor_disabled_v<int>);
  ASSERT_TRUE(is_destructor_disabled_v<std::string_view>);

  // ThreadCachedArena thread_safe_arena;
  // auto p1 = thread_safe_arena.New<arena::String>();
  // p1->assign("aaa");

  // RUDF_INFO("###cmp:{}", *p == *p1);

  // arrow::ListBuilder builder(arrow::default_memory_pool());
}