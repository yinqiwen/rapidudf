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
#include <exception>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"

using namespace rapidudf;
using namespace rapidudf::ast;

static int test_user_func() { throw std::logic_error("aaa"); }

struct TestStruct {
  void test_funcx() { throw std::logic_error("aaa"); }
};

RUDF_SAFE_FUNC_REGISTER(test_user_func, kFuncNoAttrs)

RUDF_STRUCT_SAFE_MEMBER_METHODS(TestStruct, test_funcx)

TEST(JitCompiler, exception) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
      test_user_func()
  )";
  auto rc = compiler.CompileExpression<int>(content, {}, true);
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
  auto rc = compiler.CompileExpression<int>(content, {}, true);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f.SetRethrowException(true);
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
  JitCompiler compiler;
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

RUDF_STRUCT_SAFE_HELPER_METHODS_BIND(Helper, test0, test1)

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
