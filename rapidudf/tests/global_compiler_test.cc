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
#include "absl/strings/str_join.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, func_cache) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == 123;
    }
  )";
  auto rc = GlobalJitCompiler::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));

  rc = GlobalJitCompiler::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  f = std::move(rc.value());
  ASSERT_TRUE(f(json));
  ASSERT_TRUE(f.IsFromCache());
}

TEST(JitCompiler, expr_cache) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    x["key"] == 123
  )";
  auto rc = GlobalJitCompiler::GetExpression<bool, const JsonObject&>(content, {"x"});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));

  rc = GlobalJitCompiler::GetExpression<bool, const JsonObject&>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f1 = std::move(rc.value());
  ASSERT_TRUE(f1(json));
  ASSERT_TRUE(f1.IsFromCache());
}