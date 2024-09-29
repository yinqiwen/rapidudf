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
#include <cmath>
#include <functional>
#include <vector>

#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, add) {
  JitCompiler compiler;
  std::string expr = "x+y";
  auto rc0 = compiler.CompileExpression<int, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(1, 2), 3);
  ASSERT_EQ(f0(111, 222), 333);
  auto rc1 = compiler.CompileExpression<long double, long double, long double>(expr, {"x", "y"}, true);
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  long double x = 1.1;
  long double y = 2.2;
  long double r = f1(x, y);
  ASSERT_DOUBLE_EQ(r, 3.3);
  r = f1(111.1, 222.2);
  ASSERT_DOUBLE_EQ(r, 333.3);
}
TEST(JitCompiler, vector_add) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x+y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] + i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] + f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, sub) {
  JitCompiler compiler;
  std::string expr = "x-y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  auto rc0 = compiler.CompileExpression<double, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left - f_right);

  auto rc1 = compiler.CompileExpression<int, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left - i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, vector_sub) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x-y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] - i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] - f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, multiply) {
  JitCompiler compiler;
  std::string expr = "x*y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  auto rc0 = compiler.CompileExpression<double, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left * f_right);

  auto rc1 = compiler.CompileExpression<int, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left * i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, vector_mul) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x*y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] * i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] * f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, divid) {
  JitCompiler compiler;
  std::string expr = "x/y";
  int i_left = 100, i_right = 121;
  double f_left = 30000.14, f_right = 12421.4;
  auto rc0 = compiler.CompileExpression<double, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), f_left / f_right);

  auto rc1 = compiler.CompileExpression<int, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left / i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, vector_div) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x/y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] / i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], f_left[i] / f_right[i]);
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, mod) {
  JitCompiler compiler;
  std::string expr = "x%y";
  int i_left = 100, i_right = 121;
  // double f_left = 30000.14, f_right = 12421.4;
  auto rc0 = compiler.CompileExpression<double, double, double>(expr, {"x", "y"});
  ASSERT_FALSE(rc0.ok());

  auto rc1 = compiler.CompileExpression<int, int, int>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), i_left % i_right);

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());
}

TEST(JitCompiler, vector_mod) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -20, 30, -40, 40, 80, -10};
  // std::vector<float> f_left{1, -2.1, 3.2, -4, 4, 8.66, -1, 6.7};
  // std::vector<float> f_right{1, -2.1, 3.2, -4, 4, 8.457, -1, 2.2};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x%y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, i_left, i_right);
  ASSERT_EQ(result0.Size(), i_left.size());
  for (size_t i = 0; i < result0.Size(); i++) {
    ASSERT_EQ(result0[i], i_left[i] % i_right[i]);
  }
  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_FALSE(rc1.ok());

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, pow) {
  JitCompiler compiler;
  std::string expr = "x^y";
  int64_t i_left = 100, i_right = 6;
  double f_left = 30000.14, f_right = 5.2;
  auto rc0 = compiler.CompileExpression<double, double, double>(expr, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_DOUBLE_EQ(f0(f_left, f_right), std::pow(f_left, f_right));

  auto rc1 = compiler.CompileExpression<int64_t, int64_t, int64_t>(expr, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(i_left, i_right), std::pow(i_left, i_right));

  auto rc2 = compiler.CompileExpression<Bit, Bit, Bit>(expr, {"x", "y"});
  ASSERT_FALSE(rc2.ok());

  std::string multiple_pow =
      "(Click^10.0)*((Like+0.000082)^4.7)*(Inter^3.5)*((Join+0.000024)^5.5)*(TimeV1^7.0)*((PostComment+0.000024)^3.5)*("
      "(PositiveCommentV1+0.0038)^1.0)*(ExpoTimeV1^1.5)";
  auto rc3 = compiler.CompileExpression<double, double, double, double, double, double, double, double, double>(
      multiple_pow, {"Click", "Like", "Inter", "Join", "TimeV1", "PostComment", "PositiveCommentV1", "ExpoTimeV1"});
  ASSERT_TRUE(rc3.ok());
  double Click = 10;
  double Like = 20;
  double Inter = 2;
  double Join = 3;
  double TimeV1 = 11;
  double PostComment = 12;
  double PositiveCommentV1 = 3;
  double ExpoTimeV1 = 4;
  auto f3 = std::move(rc3.value());
  double f_result = f3(Click, Like, Inter, Join, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1);
  double actual = std::pow(Click, 10.0) * std::pow(Like + 0.000082, 4.7) * std::pow(Inter, 3.5) *
                  std::pow(Join + 0.000024, 5.5) * std::pow(TimeV1, 7.0) * std::pow(PostComment + 0.000024, 3.5) *
                  std::pow(PositiveCommentV1 + 0.0038, 1.0) * std::pow(ExpoTimeV1, 1.5);
  ASSERT_DOUBLE_EQ(f_result, actual);
}

TEST(JitCompiler, vector_pow) {
  std::vector<int> i_left{1, -2, 3, -4, 4, 8, -1};
  std::vector<int> i_right{10, -10, 2, -4, 4, 8, -2};
  std::vector<float> f_left{1, 2.1, 3.2, 4, 4, 8.66, 1, 6.7};
  std::vector<float> f_right{1.1, 2.1, 3.1, 4.1, 4.1, 2.7, 1.2, 2.4};

  Context ctx;
  JitCompiler compiler;
  std::string expr = "x^y";
  auto rc0 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>>(
      expr, {"_", "x", "y"});
  ASSERT_FALSE(rc0.ok());

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, f_left, f_right);
  ASSERT_EQ(result1.Size(), f_left.size());
  for (size_t i = 0; i < result1.Size(); i++) {
    ASSERT_EQ(result1[i], std::pow(f_left[i], f_right[i]));
  }

  auto rc2 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  if (!rc2.ok()) {
    RUDF_ERROR("{}", rc2.status().ToString());
  }
  ASSERT_FALSE(rc2.ok());
  auto rc3 = compiler.CompileExpression<simd::Vector<StringView>, Context&, simd::Vector<StringView>,
                                        simd::Vector<StringView>>(expr, {"_", "x", "y"});
  if (!rc3.ok()) {
    RUDF_ERROR("{}", rc3.status().ToString());
  }
  ASSERT_FALSE(rc3.ok());
}

TEST(JitCompiler, pow_mul) {
  JitCompiler compiler;
  std::string content = "2*x^y";
  auto rc1 = compiler.CompileExpression<float, float, float>(content, {"x", "y"}, true);
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_FLOAT_EQ(f1(3, 2), 2 * std::pow(3, 2));
  ASSERT_FLOAT_EQ(f1(7, 5), 2 * std::pow(7, 5));
}
