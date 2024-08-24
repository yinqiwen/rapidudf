/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/struct_access.h"

using namespace rapidudf;
using namespace rapidudf::ast;
struct TestInternal {
  int a = 0;
};
RUDF_STRUCT_FIELDS(TestInternal, a)
struct TestStruct {
  int a = 0;
  TestInternal internal;
  std::vector<int> vec;
};

RUDF_STRUCT_FIELDS(TestStruct, internal, a, vec)

TEST(JitCompiler, struct_access) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct t;
  t.a = 101;
  t.internal.a = 102;
  JitCompiler compiler;
  std::string source = R"(
    int test_func(TestStruct x){
      return x.internal.a;
    }
  )";
  auto rc = compiler.CompileFunction<int, const TestStruct&>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(t), t.internal.a);
  t.a = 1200;
  ASSERT_EQ(f(t), t.internal.a);
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
  // auto f = compiler.GetFunc<int, const TestB*>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&t), t.base->a);
}

TEST(JitCompiler, struct_write) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct t;
  t.a = 101;
  t.internal.a = 102;
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(TestStruct x){
      x.internal.a = 105;
      return x.internal.a;
    }
  )";
  auto rc = compiler.CompileFunction<int, TestStruct&>(content);
  ASSERT_FALSE(rc.ok());
}
