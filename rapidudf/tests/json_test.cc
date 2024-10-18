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
#include <functional>
#include <vector>
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, json_read_int) {
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
  auto rc = compiler.CompileFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, const JsonObject&>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}
TEST(JitCompiler, json_read_str) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;

  JsonObject json;
  json["key"] = "hello,world";

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == "hello,world";
    }
  )";
  auto rc = compiler.CompileFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, const JsonObject&>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}

TEST(JitCompiler, json_read_float) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;

  JsonObject json;
  json["key"] = 3.14;

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == 3.14;
    }
  )";
  auto rc = compiler.CompileFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, const JsonObject&>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}

TEST(JitCompiler, json_array_get) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;

  JsonObject json;
  json["key"] = {1, 2, 3};

  std::string content = R"(
    bool test_func(json x){
      return x["key"][1] == 2;
    }
  )";
  auto rc = compiler.CompileFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, const JsonObject&>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}