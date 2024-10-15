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

#include <gtest/gtest.h>
#include <vector>

#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;

TEST(JitCompiler, vector_access) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  auto result = compiler.CompileExpression<int, std::vector<int>&>("x.size()", {"x"});
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  ASSERT_EQ(f(vec), vec.size());

  auto result1 = compiler.CompileExpression<int, std::vector<int>&>("x.find(2)", {"x"});
  ASSERT_TRUE(result1.ok());
  auto f1 = std::move(result1.value());
  ASSERT_EQ(f1(vec), 1);

  auto result2 = compiler.CompileExpression<bool, std::vector<int>&>("x.contains(4)", {"x"});
  ASSERT_TRUE(result2.ok());
  auto f2 = std::move(result2.value());
  ASSERT_EQ(f2(vec), false);

  std::vector<std::string> str_vec{"hello", "world"};
  auto str_f_result = compiler.CompileExpression<StringView, std::vector<std::string>&>("x.get(1)", {"x"});
  ASSERT_TRUE(str_f_result.ok());
  auto str_f = std::move(str_f_result.value());
  ASSERT_EQ(str_f(str_vec), "world");

  str_f_result = compiler.CompileExpression<StringView, std::vector<std::string>&>("x[1]", {"x"});
  ASSERT_TRUE(str_f_result.ok());
  str_f = std::move(str_f_result.value());
  ASSERT_EQ(str_f(str_vec), "world");

  str_f_result = compiler.CompileExpression<StringView, std::vector<std::string>&>(R"(x["t1"])", {"x"});
  ASSERT_FALSE(str_f_result.ok());
}

TEST(JitCompiler, map_access) {
  spdlog::set_level(spdlog::level::debug);
  std::map<std::string, std::string> map{{"t0", "v0"}, {"t1", "v1"}};
  JitCompiler compiler;
  auto result = compiler.CompileExpression<int, std::map<std::string, std::string>&>("x.size()", {"x"});
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  ASSERT_EQ(f(map), map.size());

  auto get_f_result =
      compiler.CompileExpression<StringView, std::map<std::string, std::string>&>(R"(x.get("t1"))", {"x"});
  ASSERT_TRUE(get_f_result.ok());
  auto str_f = std::move(get_f_result.value());
  ASSERT_EQ(str_f(map), "v1");

  auto result2 = compiler.CompileExpression<bool, std::map<std::string, std::string>&>(R"(x.contains("t1"))", {"x"});
  ASSERT_TRUE(result2.ok());
  auto f2 = std::move(result2.value());
  ASSERT_EQ(f2(map), true);

  get_f_result = compiler.CompileExpression<StringView, std::map<std::string, std::string>&>(R"(x["t1"])", {"x"});
  ASSERT_TRUE(get_f_result.ok());
  str_f = std::move(get_f_result.value());
  ASSERT_EQ(str_f(map), "v1");
}

TEST(JitCompiler, map_vector_access) {
  spdlog::set_level(spdlog::level::debug);
  std::map<std::string, std::vector<std::string>> map{{"t0", {"v00", "v01"}}, {"t1", {"v10", "v11"}}};
  JitCompiler compiler;
  reflect::register_stl_collection_member_funcs<std::map<std::string, std::vector<std::string>>>();
  auto result = compiler.CompileExpression<int, std::map<std::string, std::vector<std::string>>&>("x.size()", {"x"});
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  ASSERT_EQ(f(map), map.size());

  auto get_f_result =
      compiler.CompileExpression<StringView, std::map<std::string, std::vector<std::string>>&>(R"(x["t1"][1])", {"x"});
  ASSERT_TRUE(get_f_result.ok());
  auto str_f = std::move(get_f_result.value());
  ASSERT_EQ(str_f(map), "v11");
}

TEST(JitCompiler, unordered_map_access) {
  spdlog::set_level(spdlog::level::debug);
  std::unordered_map<std::string, std::string> map{{"t0", "v0"}, {"t1", "v1"}};
  JitCompiler compiler;
  auto result = compiler.CompileExpression<int, std::unordered_map<std::string, std::string>&>("x.size()", {"x"});
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  ASSERT_EQ(f(map), map.size());

  auto get_f_result =
      compiler.CompileExpression<StringView, std::unordered_map<std::string, std::string>&>(R"(x.get("t1"))", {"x"});
  ASSERT_TRUE(get_f_result.ok());
  auto str_f = std::move(get_f_result.value());
  ASSERT_EQ(str_f(map), "v1");

  get_f_result =
      compiler.CompileExpression<StringView, std::unordered_map<std::string, std::string>&>(R"(x["t1"])", {"x"});
  ASSERT_TRUE(get_f_result.ok());
  str_f = std::move(get_f_result.value());
  ASSERT_EQ(str_f(map), "v1");
}