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

TEST(JitCompiler, ternary) {
  JitCompiler compiler;
  std::string content = "x>3?1:0";

  auto rc = compiler.CompileExpression<int, int>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(3), 0);
  ASSERT_EQ(f(4), 1);
  ASSERT_EQ(f(6), 1);
  ASSERT_EQ(f(2), 0);
}

TEST(JitCompiler, vector_ternary) {
  JitCompiler compiler;
  Context ctx;
  std::string content = "x>3?1:0";
  std::vector<int> cond_var = {1, 2, 1, 2, 4, 1, 4, 5, 6};
  auto rc = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, cond_var);
  ASSERT_EQ(result.Size(), cond_var.size());
  for (size_t i = 0; i < cond_var.size(); i++) {
    ASSERT_EQ(result[i], cond_var[i] > 3 ? 1 : 0);
  }
}