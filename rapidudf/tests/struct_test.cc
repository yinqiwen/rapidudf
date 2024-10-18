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
#include "rapidudf/compiler/options.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;
struct TestInternal {
  int a = 0;
};
RUDF_STRUCT_FIELDS(TestInternal, a)
struct TestStruct {
  int a = 0;
  TestInternal internal;
  TestInternal* internal_ptr = nullptr;
  std::vector<int> vec;
};

RUDF_STRUCT_FIELDS(TestStruct, internal, internal_ptr, a, vec)

TEST(JitCompiler, struct_access) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct t;
  t.a = 101;
  t.internal.a = 102;
  rapidudf::compiler::Options opt;
  opt.optimize_level = 0;
  JitCompiler compiler(opt);
  std::string source = "x.internal.a";
  auto rc = compiler.CompileExpression<int, const TestStruct&>(source, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(t), t.internal.a);
  t.a = 1200;
  ASSERT_EQ(f(t), t.internal.a);

  TestInternal tmp;
  t.internal_ptr = &tmp;
  std::string ptr_source = "x.internal_ptr.a";
  rc = compiler.CompileExpression<int, const TestStruct&>(ptr_source, {"x"});
  ASSERT_TRUE(rc.ok());
  f = std::move(rc.value());
  ASSERT_EQ(f(t), t.internal_ptr->a);
  tmp.a = 1200;
  ASSERT_EQ(f(t), t.internal_ptr->a);
}

TEST(JitCompiler, return_cast) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct t;
  t.a = 101;
  t.internal.a = 102;
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    i64 test_func(TestStruct x){
      return x.a;
    }
  )";
  auto rc = compiler.CompileFunction<int64_t, const TestStruct*>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int64_t, const TestA*>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&t), t.a);
  t.a = 1200;
  ASSERT_EQ(f(&t), t.a);
}

struct TestB {
  int a = 0;
  TestInternal* base;
};

RUDF_STRUCT_FIELDS(TestB, base, a)
TEST(JitCompiler, struct_access_ptr) {
  spdlog::set_level(spdlog::level::debug);
  TestB t;
  t.a = 101;
  t.base = new TestInternal;
  t.base->a = 102;
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(TestB x){
      return x.base.a;
    }
  )";
  auto rc = compiler.CompileFunction<int, const TestB*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&t), t.base->a);
}

TEST(JitCompiler, struct_write) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct t;
  t.a = 101;
  t.internal.a = 102;
  JitCompiler compiler({.fast_math = false});
  ParseContext ctx;
  std::string content = R"(
    int test_func(TestStruct x){
      x.internal.a = 105;
      return x.internal.a;
    }
  )";
  auto rc = compiler.CompileFunction<int, TestStruct&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(t), t.internal.a);
  ASSERT_EQ(105, t.internal.a);
}