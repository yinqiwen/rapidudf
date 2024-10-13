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
  auto x0 = ctx.NewSimdVector(bvs0);
  auto x1 = ctx.NewSimdVector(bvs1);
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>>(expr, {"_", "x"});
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
  auto b_left = ctx.NewSimdVector(left);
  auto b_right = ctx.NewSimdVector(right);
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
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
  auto b_left = ctx.NewSimdVector(left);
  auto b_right = ctx.NewSimdVector(right);
  auto rc0 = compiler.CompileExpression<simd::Vector<Bit>, Context&, simd::Vector<Bit>, simd::Vector<Bit>>(
      expr, {"_", "x", "y"});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  auto result0 = f0(ctx, b_left, b_right);
  ASSERT_EQ(result0.Size(), b_left.Size());
  for (size_t i = 0; i < left.size(); i++) {
    ASSERT_EQ(result0[i], left[i] || right[i]);
  }
}