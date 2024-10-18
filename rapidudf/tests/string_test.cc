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
#include <vector>

#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, string_size) {
  JitCompiler compiler;
  std::string content = R"(str.size())";
  auto rc = compiler.CompileExpression<size_t, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), 5);
  ASSERT_EQ(f("he"), 2);
}
TEST(JitCompiler, string_contains) {
  JitCompiler compiler;
  std::string content = R"(str.contains("hello"))";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), true);
  ASSERT_EQ(f("he"), false);
}
TEST(JitCompiler, string_starts_with) {
  JitCompiler compiler;
  std::string content = R"(str.starts_with("hello"))";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), true);
  ASSERT_EQ(f("he"), false);
}