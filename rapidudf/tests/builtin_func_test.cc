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
#include <functional>
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"
using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, abs) {
  JitCompiler compiler;
  std::string content = "abs(x)";
  auto rc0 = compiler.CompileExpression<int64_t, int64_t>(content, {"x"});
  if (!rc0.ok()) {
    RUDF_ERROR("{}", rc0.status().ToString());
  }
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(-11), 11);
  ASSERT_EQ(f0(0), 0);
  ASSERT_EQ(f0(2), 2);
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(-11.1), 11.1);
  ASSERT_DOUBLE_EQ(f1(0), 0);
  ASSERT_DOUBLE_EQ(f1(2.2), 2.2);
  auto rc2 = compiler.CompileExpression<float, float>(content, {"x"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  ASSERT_FLOAT_EQ(f2(-11.1), 11.1);
  ASSERT_FLOAT_EQ(f2(0), 0);
  ASSERT_FLOAT_EQ(f2(2.2), 2.2);
}

TEST(JitCompiler, vector_abs) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "abs(x)";
  std::vector<int64_t> ivs{1, -2, 3, -4, 4, 8, -1};
  auto rc4 = compiler.CompileExpression<simd::Vector<int64_t>, Context&, simd::Vector<int64_t>>(content, {"_", "x"});
  if (!rc4.ok()) {
    RUDF_ERROR("{}", rc4.status().ToString());
  }
  ASSERT_TRUE(rc4.ok());
  auto f4 = std::move(rc4.value());
  auto result4 = f4(ctx, ivs);
  ASSERT_EQ(result4.Size(), ivs.size());
  for (size_t i = 0; i < ivs.size(); i++) {
    ASSERT_EQ(result4[i], std::abs(ivs[i]));
  }

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::abs(fvs[i]));
  }
}

TEST(JitCompiler, max) {
  JitCompiler compiler;
  std::string content = "max(x,y)";
  auto rc0 = compiler.CompileExpression<int64_t, int64_t, int64_t>(content, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(-11, 12), 12);
  ASSERT_EQ(f0(0, 0), 0);
  ASSERT_EQ(f0(2, 1), 2);
  auto rc1 = compiler.CompileExpression<double, double, double>(content, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(-11.1, 11.2), 11.2);
  ASSERT_DOUBLE_EQ(f1(0, 0), 0);
  ASSERT_DOUBLE_EQ(f1(2.2, 2.1), 2.2);
  auto rc2 = compiler.CompileExpression<float, float, float>(content, {"x", "y"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  ASSERT_FLOAT_EQ(f2(-11.1, 12.2), 12.2);
  ASSERT_FLOAT_EQ(f2(0, 0), 0);
  ASSERT_FLOAT_EQ(f2(2.2, 1.1), 2.2);

  auto rc3 = compiler.CompileExpression<uint64_t, uint64_t, uint64_t>(content, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(11, 12), 12);
  ASSERT_EQ(f3(0, 0), 0);
  ASSERT_EQ(f3(2, 1), 2);
}

TEST(JitCompiler, vector_max) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "max(x,y)";

  std::vector<int64_t> x{1, -2, 3, -4, 4, 8, -1};
  std::vector<int64_t> y{-1, -2, 13, 14, -14, -8, 1};
  std::vector<double> fx{1.1, -2.1, 3.2, -4.1, 4.2, 8.1, -1.12};
  std::vector<double> fy{1.13, -2.12, 3.02, -4.01, 4.01, 8.02, -1.0001};
  auto rc0 = compiler.CompileExpression<simd::Vector<int64_t>, Context&, simd::Vector<int64_t>, simd::Vector<int64_t>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, x, y);
  ASSERT_EQ(result0.Size(), x.size());
  for (size_t i = 0; i < x.size(); i++) {
    ASSERT_FLOAT_EQ(result0[i], std::max(x[i], y[i]));
  }

  auto rc1 = compiler.CompileExpression<simd::Vector<double>, Context&, simd::Vector<double>, simd::Vector<double>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy);
  ASSERT_EQ(result1.Size(), x.size());
  for (size_t i = 0; i < x.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::max(fx[i], fy[i]));
  }
}

TEST(JitCompiler, min) {
  JitCompiler compiler;
  std::string content = "min(x,y)";
  auto rc0 = compiler.CompileExpression<int64_t, int64_t, int64_t>(content, {"x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(-11, 12), -11);
  ASSERT_EQ(f0(0, 0), 0);
  ASSERT_EQ(f0(2, 1), 1);
  auto rc1 = compiler.CompileExpression<double, double, double>(content, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(-11.1, 11.2), -11.1);
  ASSERT_DOUBLE_EQ(f1(0, 0), 0);
  ASSERT_DOUBLE_EQ(f1(2.2, 2.1), 2.1);
  auto rc2 = compiler.CompileExpression<float, float, float>(content, {"x", "y"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  ASSERT_FLOAT_EQ(f2(-11.1, 12.2), -11.1);
  ASSERT_FLOAT_EQ(f2(0, 0), 0);
  ASSERT_FLOAT_EQ(f2(2.2, 1.1), 1.1);

  auto rc3 = compiler.CompileExpression<uint64_t, uint64_t, uint64_t>(content, {"x", "y"});
  ASSERT_TRUE(rc3.ok());
  auto f3 = std::move(rc3.value());
  ASSERT_EQ(f3(11, 12), 11);
  ASSERT_EQ(f3(0, 0), 0);
  ASSERT_EQ(f3(2, 1), 1);
}

TEST(JitCompiler, vector_min) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "min(x,y)";

  std::vector<int32_t> x{1, -2, 3, -4, 4, 8, -1};
  std::vector<int32_t> y{-1, -2, 13, 14, -14, -8, 1};
  std::vector<float> fx{1.1, -2.1, 3.2, -4.1, 4.2, 8.1, -1.12};
  std::vector<float> fy{1.13, -2.12, 3.02, -4.01, 4.01, 8.02, -1.0001};
  auto rc0 = compiler.CompileExpression<simd::Vector<int32_t>, Context&, simd::Vector<int32_t>, simd::Vector<int32_t>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, x, y);
  ASSERT_EQ(result0.Size(), x.size());
  for (size_t i = 0; i < x.size(); i++) {
    ASSERT_FLOAT_EQ(result0[i], std::min(x[i], y[i]));
  }

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy);
  ASSERT_EQ(result1.Size(), x.size());
  for (size_t i = 0; i < x.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::min(fx[i], fy[i]));
  }
}

TEST(JitCompiler, ceil) {
  JitCompiler compiler;
  std::string content = "ceil(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::ceil(x));
}

TEST(JitCompiler, vector_ceil) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "ceil(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::ceil(fvs[i]));
  }
}

TEST(JitCompiler, erf) {
  JitCompiler compiler;
  std::string content = "erf(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::erf(x));
}

TEST(JitCompiler, vector_erf) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "erf(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::erf(fvs[i]));
  }
}

TEST(JitCompiler, erfc) {
  JitCompiler compiler;
  std::string content = "erfc(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::erfc(x));
}

TEST(JitCompiler, vector_erfc) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "erfc(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::erfc(fvs[i]));
  }
}

TEST(JitCompiler, exp) {
  JitCompiler compiler;
  std::string content = "exp(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::exp(x));
}

TEST(JitCompiler, vector_exp) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "exp(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::exp(fvs[i]));
  }
}

TEST(JitCompiler, exp2) {
  JitCompiler compiler;
  std::string content = "exp2(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::exp2(x));
}

TEST(JitCompiler, vector_exp2) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "exp2(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::exp2(fvs[i]));
  }
}

TEST(JitCompiler, expm1) {
  JitCompiler compiler;
  std::string content = "expm1(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::expm1(x));
}

TEST(JitCompiler, vector_expm1) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "expm1(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::expm1(fvs[i]));
  }
}

TEST(JitCompiler, floor) {
  JitCompiler compiler;
  std::string content = "floor(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::floor(x));
}

TEST(JitCompiler, vector_floor) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "floor(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::floor(fvs[i]));
  }
}
TEST(JitCompiler, round) {
  JitCompiler compiler;
  std::string content = "round(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::round(x));
}

TEST(JitCompiler, vector_round) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "round(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::round(fvs[i]));
  }
}

TEST(JitCompiler, rint) {
  JitCompiler compiler;
  std::string content = "rint(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::rint(x));
}

TEST(JitCompiler, vector_rint) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "rint(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::rint(fvs[i]));
  }
}

TEST(JitCompiler, trunc) {
  JitCompiler compiler;
  std::string content = "trunc(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::trunc(x));
}

TEST(JitCompiler, vector_trunc) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "trunc(x)";

  std::vector<float> fvs{1, -2.1, 3.2, -4, 4, 8, -1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::trunc(fvs[i]));
  }
}

TEST(JitCompiler, sqrt) {
  JitCompiler compiler;
  std::string content = "sqrt(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::sqrt(x));
}

TEST(JitCompiler, vector_sqrt) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "sqrt(x)";

  std::vector<float> fvs{1.2, 2.1, 3.2, 1.2, 4, 8, 1.5};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::sqrt(fvs[i]));
  }
}

TEST(JitCompiler, log) {
  JitCompiler compiler;
  std::string content = "log(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::log(x));
}

TEST(JitCompiler, vector_log) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "log(x)";

  std::vector<double> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<double>, Context&, simd::Vector<double>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::log(fvs[i]));
  }
}

TEST(JitCompiler, log2) {
  JitCompiler compiler;
  std::string content = "log2(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::log2(x));
}

TEST(JitCompiler, vector_log2) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "log2(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::log2(fvs[i]));
  }
}

TEST(JitCompiler, log10) {
  JitCompiler compiler;
  std::string content = "log10(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::log10(x));
}

TEST(JitCompiler, vector_log10) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "log10(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::log10(fvs[i]));
  }
}

TEST(JitCompiler, log1p) {
  JitCompiler compiler;
  std::string content = "log1p(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::log1p(x));
}

TEST(JitCompiler, vector_log1p) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "log1p(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::log1p(fvs[i]));
  }
}

TEST(JitCompiler, sin) {
  JitCompiler compiler;
  std::string content = "sin(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::sin(x));
}

TEST(JitCompiler, vector_sin) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "sin(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::sin(fvs[i]));
  }
}

TEST(JitCompiler, cos) {
  JitCompiler compiler;
  std::string content = "cos(x)";
  long double x = 3.14;
  auto rc1 = compiler.CompileExpression<long double, long double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::cos(x));
}

TEST(JitCompiler, vector_cos) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "cos(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::cos(fvs[i]));
  }
}

TEST(JitCompiler, tan) {
  JitCompiler compiler;
  std::string content = "tan(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::tan(x));
}

TEST(JitCompiler, vector_tan) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "tan(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::tan(fvs[i]));
  }
}

TEST(JitCompiler, sinh) {
  JitCompiler compiler;
  std::string content = "sinh(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::sinh(x));
}

TEST(JitCompiler, vector_sinh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "sinh(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::sinh(fvs[i]));
  }
}

TEST(JitCompiler, cosh) {
  JitCompiler compiler;
  std::string content = "cosh(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::cosh(x));
}

TEST(JitCompiler, vector_cosh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "cosh(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::cosh(fvs[i]));
  }
}

TEST(JitCompiler, tanh) {
  JitCompiler compiler;
  std::string content = "tanh(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::tanh(x));
}

TEST(JitCompiler, vector_tanh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "tanh(x)";

  std::vector<float> fvs{0.3, 0.5, 3.2, 0.6, 4, 8, 1.1};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::tanh(fvs[i]));
  }
}

TEST(JitCompiler, asin) {
  JitCompiler compiler;
  std::string content = "asin(x)";
  double x = 0.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::asin(x));
}

TEST(JitCompiler, vector_asin) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "asin(x)";

  std::vector<float> fvs{0.3, 0.5, 0.2, 0.6, 0.7, 0, 1.0};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::asin(fvs[i]));
  }
}
// TEST(JitCompiler, column_asin) {
//   Context ctx;
//   JitCompiler compiler;
//   std::string content = "asin(x)";
//   std::vector<float> fvs{0.3, 0.5, 0.2, 0.6, 0.7, 0, 1.0};
//   auto& ctx_ = ctx;
//   simd::Column* column = ctx.New<simd::Column>(ctx_, fvs);
//   simd::Table table(ctx);

//   auto rc5 = compiler.CompileExpression<simd::Column*, Context&, simd::Column*>(content, {"_", "x"});
//   if (!rc5.ok()) {
//     RUDF_ERROR("{}", rc5.status().ToString());
//   }
//   ASSERT_TRUE(rc5.ok());
//   auto f5 = std::move(rc5.value());
//   auto result5 = f5(ctx, column);
//   ASSERT_EQ(result5->size(), fvs.size());
//   auto result_vec = result5->ToVector<float>().value();
//   for (size_t i = 0; i < fvs.size(); i++) {
//     ASSERT_FLOAT_EQ(result_vec[i], std::asin(fvs[i]));
//   }
// }

TEST(JitCompiler, acos) {
  JitCompiler compiler;
  std::string content = "acos(x)";
  double x = 0.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::acos(x));
}

TEST(JitCompiler, vector_acos) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "acos(x)";

  std::vector<float> fvs{0.3, 0.5, 0.2, 0.6, 0.7, 0, 1.0};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::acos(fvs[i]));
  }
}

TEST(JitCompiler, atan) {
  JitCompiler compiler;
  std::string content = "atan(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::atan(x));
}

TEST(JitCompiler, vector_atan) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "atan(x)";

  std::vector<float> fvs{0.3, 0.5, 0.2, 0.6, 0.7, 0, 1.0};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::atan(fvs[i]));
  }
}

TEST(JitCompiler, asinh) {
  JitCompiler compiler;
  std::string content = "asinh(x)";
  long double x = 3.14;
  auto rc1 = compiler.CompileExpression<long double, long double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::asinh(x));
}

TEST(JitCompiler, vector_asinh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "asinh(x)";

  std::vector<float> fvs{0.3, 0.5, 0.2, 0.6, 0.7, 0, 1.0};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::asinh(fvs[i]));
  }
}

TEST(JitCompiler, acosh) {
  JitCompiler compiler;
  std::string content = "acosh(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::acosh(x));
}

TEST(JitCompiler, vector_acosh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "acosh(x)";

  std::vector<float> fvs{10.1, 2.5, 1.2, 2.6, 3.7, 4.6, 7.0};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::acosh(fvs[i]));
  }
}

TEST(JitCompiler, atanh) {
  JitCompiler compiler;
  std::string content = "atanh(x)";
  double x = 0.9;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::atanh(x));
}

TEST(JitCompiler, vector_atanh) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "atanh(x)";

  std::vector<float> fvs{0.8, 0.5, 0.2, 0.6, 0.7, 0.6, 0.7};
  auto rc5 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>>(content, {"_", "x"});
  if (!rc5.ok()) {
    RUDF_ERROR("{}", rc5.status().ToString());
  }
  ASSERT_TRUE(rc5.ok());
  auto f5 = std::move(rc5.value());
  auto result5 = f5(ctx, fvs);
  ASSERT_EQ(result5.Size(), fvs.size());
  for (size_t i = 0; i < fvs.size(); i++) {
    ASSERT_FLOAT_EQ(result5[i], std::atanh(fvs[i]));
  }
}

TEST(JitCompiler, atan2) {
  JitCompiler compiler;
  std::string content = "atan2(x,y)";
  double x = 0.9, y = -0.5;
  auto rc1 = compiler.CompileExpression<double, double, double>(content, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x, y), std::atan2(x, y));
}

TEST(JitCompiler, vector_atan2) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "atan2(x,y)";

  std::vector<float> fx{1.1, -2.1, 3.2, -3.1, 2.2, 1.1, -1.12};
  std::vector<float> fy{1.13, -2.12, 3.02, -2.01, 2.01, 2.02, -1.0001};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::atan2(fx[i], fy[i]));
  }
}

TEST(JitCompiler, hypot) {
  JitCompiler compiler;
  std::string content = "hypot(x,y)";
  double x = 0.9, y = -0.5;
  auto rc1 = compiler.CompileExpression<double, double, double>(content, {"x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x, y), std::hypot(x, y));
}

TEST(JitCompiler, vector_hypot) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "hypot(x,y)";

  std::vector<float> fx{1.1, -2.1, 3.2, -3.1, 2.2, 1.1, -1.12};
  std::vector<float> fy{1.13, -2.12, 3.02, -2.01, 2.01, 2.02, -1.0001};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>>(
      content, {"_", "x", "y"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::hypot(fx[i], fy[i]));
  }
}

TEST(JitCompiler, vector_sum) {
  std::vector<float> left{1, 2, 3, 4, 1, 5, 6};
  std::vector<float> right{10, 20, 30, 40, 10, 50, 60};
  simd::Vector<float> simd_left(left);
  simd::Vector<float> simd_right(right);
  JitCompiler compiler;
  std::string content = R"(
    f32 test_func(simd_vector<f32> x){
      return sum(x);
    }
  )";
  auto rc = compiler.CompileFunction<float, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_left);
  float native_result = 0;
  for (size_t i = 0; i < left.size(); i++) {
    native_result += (left[i]);
  }
  ASSERT_FLOAT_EQ(result, native_result);
}

TEST(JitCompiler, vector_dot) {
  std::vector<float> left{1, 2, 3, 4, 1, 5, 6};
  std::vector<float> right{10, 20, 30, 40, 10, 50, 60};
  simd::Vector<float> simd_left(left);
  simd::Vector<float> simd_right(right);
  JitCompiler compiler;
  std::string content = R"(
    f32 test_func(simd_vector<f32> x,simd_vector<f32> y){
      return dot(x,y);
    }
  )";
  auto rc = compiler.CompileFunction<float, simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_left, simd_right);
  float native_result = 0;
  for (size_t i = 0; i < left.size(); i++) {
    native_result += (left[i] * right[i]);
  }
  ASSERT_FLOAT_EQ(result, native_result);
}

TEST(JitCompiler, vector_iota) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f64> test_func(Context ctx){
      auto t = iota(1_f64,12);
      return t;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<double>, Context&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx);
  RUDF_INFO("IsReadonly:{}", result.IsReadonly());
  ASSERT_EQ(result.Size(), 12);
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_DOUBLE_EQ(result[i], i + 1);
  }
}

TEST(JitCompiler, fma) {
  JitCompiler compiler;
  std::string content = "fma(x,y,z)";
  float fx = 3.1, fy = 3.2, fz = 1.3;
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(fx, fy, fz), std::fma(fx, fy, fz));
}

TEST(JitCompiler, vector_fma) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "fma(x,y,z)";
  std::vector<float> fx{-1.1};
  std::vector<float> fy{-1.13};
  float fz = {3.5};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>, float>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy, fz);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::fma(fx[i], fy[i], fz));
  }

  std::vector<int32_t> ix{-1, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> iy{-1, -1, -2, -3, -4, -5, -6};
  int32_t iz = 11;

  auto rc2 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>, int>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  auto result2 = f2(ctx, ix, iy, iz);
  ASSERT_EQ(result2.Size(), ix.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_EQ(result2[i], ix[i] * iy[i] + iz);
  }
}

TEST(JitCompiler, clamp) {
  JitCompiler compiler;
  std::string content = "clamp(x,y,z)";
  float fx = 3.1, fy = 3.2, fz = 1.3;
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(fx, fy, fz), std::clamp(fx, fy, fz));
}

TEST(JitCompiler, vector_clamp) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "clamp(x,y,z)";
  std::vector<float> fx{-1.1};
  std::vector<float> fy{-1.13};
  float fz = {3.5};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>, float>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy, fz);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], std::clamp(fx[i], fy[i], fz));
  }

  std::vector<int32_t> ix{-1, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> iy{-1, -1, -2, -3, -4, -5, -6};
  int32_t iz = 11;

  auto rc2 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>, int>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  auto result2 = f2(ctx, ix, iy, iz);
  ASSERT_EQ(result2.Size(), ix.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_EQ(result2[i], std::clamp(ix[i], iy[i], iz));
  }
}

TEST(JitCompiler, fms) {
  JitCompiler compiler;
  std::string content = "fms(x,y,z)";
  float fx = 3.1, fy = 3.2, fz = 1.3;
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(fx, fy, fz), fx * fy - fz);
}

TEST(JitCompiler, vector_fms) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "fms(x,y,z)";
  std::vector<float> fx{-1.1};
  std::vector<float> fy{-1.13};
  float fz = {3.5};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>, float>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy, fz);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], fx[i] * fy[i] - fz);
  }

  std::vector<int32_t> ix{-1, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> iy{-1, -1, -2, -3, -4, -5, -6};
  int32_t iz = 11;

  auto rc2 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>, int>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  auto result2 = f2(ctx, ix, iy, iz);
  ASSERT_EQ(result2.Size(), ix.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_EQ(result2[i], ix[i] * iy[i] - iz);
  }
}

TEST(JitCompiler, fnma) {
  JitCompiler compiler;
  std::string content = "fnma(x,y,z)";
  float fx = 3.1, fy = 3.2, fz = 1.3;
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(fx, fy, fz), -fx * fy + fz);
}

TEST(JitCompiler, vector_fnma) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "fnma(x,y,z)";
  std::vector<float> fx{-1.1};
  std::vector<float> fy{-1.13};
  float fz = {3.5};

  auto rc1 = compiler.CompileExpression<simd::Vector<float>, Context&, simd::Vector<float>, simd::Vector<float>, float>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy, fz);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_FLOAT_EQ(result1[i], -fx[i] * fy[i] + fz);
  }

  std::vector<int32_t> ix{-1, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> iy{-1, -1, -2, -3, -4, -5, -6};
  int32_t iz = 11;

  auto rc2 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>, int>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  auto result2 = f2(ctx, ix, iy, iz);
  ASSERT_EQ(result2.Size(), ix.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_EQ(result2[i], -ix[i] * iy[i] + iz);
  }
}

TEST(JitCompiler, fnms) {
  JitCompiler compiler;
  std::string content = "fnms(x,y,z)";
  float fx = 3.1, fy = 3.2, fz = 1.3;
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(fx, fy, fz), -fx * fy - fz);
}

TEST(JitCompiler, vector_fnms) {
  Context ctx;
  JitCompiler compiler;
  std::string content = "fnms(x,y,z)";
  std::vector<double> fx{-1.1};
  std::vector<double> fy{-1.13};
  double fz = {3.5};

  auto rc1 =
      compiler.CompileExpression<simd::Vector<double>, Context&, simd::Vector<double>, simd::Vector<double>, double>(
          content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  auto result1 = f1(ctx, fx, fy, fz);
  ASSERT_EQ(result1.Size(), fx.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_DOUBLE_EQ(result1[i], -fx[i] * fy[i] - fz);
  }

  std::vector<int32_t> ix{-1, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> iy{-1, -1, -2, -3, -4, -5, -6};
  int32_t iz = 11;

  auto rc2 = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<int>, int>(
      content, {"_", "x", "y", "z"});
  ASSERT_TRUE(rc2.ok());
  auto f2 = std::move(rc2.value());
  auto result2 = f2(ctx, ix, iy, iz);
  ASSERT_EQ(result2.Size(), ix.size());
  for (size_t i = 0; i < fx.size(); i++) {
    ASSERT_EQ(result2[i], -ix[i] * iy[i] - iz);
  }
}