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
