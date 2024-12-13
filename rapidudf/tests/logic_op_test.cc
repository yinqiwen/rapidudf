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

TEST(JitCompiler, logic_not) {
  JitCompiler compiler;
  std::string expr = "!x";
  bool i_left = true, i_right = false;
  auto rc0 = compiler.CompileExpression<bool, bool>(expr, {"x"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(i_left), !i_left);
  ASSERT_EQ(f0(i_right), !i_right);

  auto rc1 = compiler.CompileExpression<bool, int>(expr, {"x"});
  ASSERT_FALSE(rc1.ok());
  auto rc2 = compiler.CompileExpression<bool, double>(expr, {"x"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView>(expr, {"x"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, logic_and) {
  JitCompiler compiler;
  std::string expr = "x&&y";
  bool i_left = true, i_right = false;
  auto rc0 = compiler.CompileExpression<bool, bool, bool>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(i_left, i_right), i_left && i_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_FALSE(rc1.ok());
  auto rc2 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, logic_or) {
  JitCompiler compiler;
  std::string expr = "x||y";
  bool i_left = true, i_right = false;
  auto rc0 = compiler.CompileExpression<bool, bool, bool>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(i_left, i_right), i_left || i_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_FALSE(rc1.ok());
  auto rc2 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, vector_logic_not) {
  JitCompiler compiler;
  Context ctx;
  std::string expr = "!x";
  std::vector<bool> bvs0 = {true, false, true, false, true, true, false, false};
  std::vector<bool> bvs1 = {true, false, true, false, true, true, false, false, false, true, true};
  auto x0 = ctx.NewVector(bvs0);
  auto x1 = ctx.NewVector(bvs1);
  auto rc0 = compiler.CompileExpression<Vector<Bit>, Context&, Vector<Bit>>(expr, {"_", "x"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, x0);
  ASSERT_EQ(result0.Size(), bvs0.size());
  for (size_t i = 0; i < bvs0.size(); i++) {
    ASSERT_EQ(result0[i], !bvs0[i]);
  }
  result0 = f0(ctx, x1);
  ASSERT_EQ(result0.Size(), bvs1.size());
  for (size_t i = 0; i < bvs0.size(); i++) {
    ASSERT_EQ(result0[i], !bvs1[i]);
  }
}

TEST(JitCompiler, vector_logic_and) {
  JitCompiler compiler;
  Context ctx;
  std::string expr = "x&&y";
  std::vector<bool> left = {true, true, true, false, false, true, false, false, false, true, true};
  std::vector<bool> right = {true, false, true, false, true, true, false, false, false, true, true};
  auto b_left = ctx.NewVector(left);
  auto b_right = ctx.NewVector(right);
  auto rc0 = compiler.CompileExpression<Vector<Bit>, Context&, Vector<Bit>, Vector<Bit>>(expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, b_left, b_right);
  ASSERT_EQ(result0.Size(), b_left.Size());
  for (size_t i = 0; i < left.size(); i++) {
    ASSERT_EQ(result0[i], left[i] && right[i]);
  }
}

TEST(JitCompiler, vector_logic_or) {
  JitCompiler compiler;
  Context ctx;
  std::string expr = "x||y";
  std::vector<bool> left = {true, true, true, false, false, true, false, false, false, true, true};
  std::vector<bool> right = {true, false, true, false, true, true, false, false, false, true, true};
  auto b_left = ctx.NewVector(left);
  auto b_right = ctx.NewVector(right);
  auto rc0 = compiler.CompileExpression<Vector<Bit>, Context&, Vector<Bit>, Vector<Bit>>(expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, b_left, b_right);
  ASSERT_EQ(result0.Size(), b_left.Size());
  for (size_t i = 0; i < left.size(); i++) {
    ASSERT_EQ(result0[i], left[i] || right[i]);
  }
}