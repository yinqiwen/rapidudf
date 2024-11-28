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
#include <exception>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;

static int test_user_func() { return 101; }

struct TestStruct {
  void test_funcx() { throw std::logic_error("aaa"); }
};

RUDF_FUNC_REGISTER(test_user_func)

RUDF_FUNC_REGISTER_WITH_NAME("NS::test_user_func", test_user_func)

TEST(JitCompiler, func) {
  JitCompiler compiler({.optimize_level = 0, .print_asm = false});

  std::string content = R"(
      test_user_func()
  )";
  auto rc = compiler.CompileExpression<int>(content, {});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());

  content = R"(
      NS::test_user_func()
  )";
  rc = compiler.CompileExpression<int>(content, {});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
}