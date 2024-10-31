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
#include <exception>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;

static int test_user_func() { throw std::logic_error("aaa"); }

struct TestStruct {
  void test_funcx() { throw std::logic_error("aaa"); }
};

RUDF_FUNC_REGISTER(test_user_func)

RUDF_STRUCT_MEMBER_METHODS(TestStruct, test_funcx)

TEST(JitCompiler, exception) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler({.optimize_level = 0, .print_asm = true});

  std::string content = R"(
      test_user_func()
  )";
  auto rc = compiler.CompileExpression<int>(content, {});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_ANY_THROW(f());
}

TEST(JitCompiler, rethrow_exception) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
      test_user_func()
  )";
  auto rc = compiler.CompileExpression<int>(content, {});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  try {
    f();
    ASSERT_TRUE(1 == 0);
  } catch (...) {
    ASSERT_TRUE(1 == 1);
  }
}

TEST(JitCompiler, member_func_exception) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler({.print_asm = true});
  std::string content = R"(
    void test_func(TestStruct x){
      x.test_funcx();
    }
   )";
  auto rc = compiler.CompileFunction<void, TestStruct*>(content);
  ASSERT_TRUE(rc.ok());
  TestStruct t;
  auto f = std::move(rc.value());
  ASSERT_ANY_THROW(f(&t));
}

struct Helper {
  static void test0(TestStruct*) {
    RUDF_DEBUG("test0");
    throw std::logic_error("test0");
  }
  static void test1(TestStruct*) {
    RUDF_DEBUG("test1");
    throw std::logic_error("test1");
  }
};

RUDF_STRUCT_HELPER_METHODS_BIND(Helper, test0, test1)

TEST(JitCompiler, member_func_bind) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  std::string content = R"(
    void test_func(TestStruct x){
      x.test0();
      x.test1();
    }
   )";
  auto rc = compiler.CompileFunction<void, TestStruct*>(content);
  ASSERT_TRUE(rc.ok());
  TestStruct t;
  auto f = std::move(rc.value());
  ASSERT_ANY_THROW(f(&t));
}