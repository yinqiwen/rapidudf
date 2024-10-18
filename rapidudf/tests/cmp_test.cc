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
#include <string_view>
#include <vector>

#include "rapidudf/rapidudf.h"

using namespace rapidudf;
TEST(JitCompiler, gt) {
  JitCompiler compiler;
  std::string expr = "x>y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left > f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left > i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());

  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left > s_right);
}

TEST(JitCompiler, ge) {
  JitCompiler compiler;
  std::string expr = "x>=y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left >= f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left >= i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());

  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left >= s_right);
}
TEST(JitCompiler, lt) {
  JitCompiler compiler;
  std::string expr = "x<y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left < f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left < i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());

  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left < s_right);
}
TEST(JitCompiler, le) {
  JitCompiler compiler;
  std::string expr = "x<=y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left <= f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left <= i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left <= s_right);
}
TEST(JitCompiler, eq) {
  JitCompiler compiler;
  std::string expr = "x==y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left == f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left == i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left == s_right);
}
TEST(JitCompiler, neq) {
  JitCompiler compiler;
  std::string expr = "x!=y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  StringView s_left = "hello", s_right = "world";
  auto rc0 = compiler.CompileExpression<bool, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left != f_right);

  auto rc1 = compiler.CompileExpression<bool, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left != i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<bool, StringView, StringView>(expr, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(s_left, s_right), s_left != s_right);
}
TEST(JitCompiler, vector_gt) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};
  Context ctx;
  JitCompiler compiler;
  std::string expr = "x>y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] > i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] > f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("###{}", rc3.status().ToString());
  }
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] > s_right[i]);
  }
}
TEST(JitCompiler, vector_ge) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x>=y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] >= i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] >= f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] >= s_right[i]);
  }
}

TEST(JitCompiler, vector_lt) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x<y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] < i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] < f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] < s_right[i]);
  }
}

TEST(JitCompiler, vector_le) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x<=y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] <= i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] <= f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] <= s_right[i]);
  }
}

TEST(JitCompiler, vector_eq) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<double> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<double> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};
  Context ctx;
  JitCompiler compiler;
  std::string expr = "x==y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] == i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<double>, simd::Vector<double>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] == f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] == s_right[i]);
  }
}

TEST(JitCompiler, vector_neq) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<double> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<double> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};
  std::vector<StringView> s_left{"s0", "s1", "s2", "s3", "s4", "s6", "s7", "s8", "s9", "s0", "s0"};
  std::vector<StringView> s_right{"a0", "b1", "c2", "e3", "d4", "f6", "87", "s8", "x9", "z0", "q0"};
  Context ctx;
  JitCompiler compiler;
  std::string expr = "x!=y";
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] != i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<double>, simd::Vector<double>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] != f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());

  auto rc3 =
      compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
          expr, {"_", "x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  auto result3 = f3(ctx, s_left, s_right);
  ASSERT_EQ(result3.Size(), s_left.size());
  for (size_t i = 0; i < result3.Size(); i++) {
    ASSERT_EQ(result3[i], s_left[i] != s_right[i]);
  }
}