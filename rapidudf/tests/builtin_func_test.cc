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
using namespace rapidudf::ast;
TEST(JitCompiler, abs) {
  JitCompiler compiler;
  std::string content = "abs(x)";
  auto rc0 = compiler.CompileExpression<int64_t, int64_t>(content, {"x"});
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

  // auto rc3 = compiler.CompileExpression<uint64_t, uint64_t>(content, {"x"});
  // ASSERT_FALSE(rc3.ok());

  Context ctx;
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

TEST(JitCompiler, fma) {
  JitCompiler compiler;
  std::string content = "fma(x,y,z)";
  auto rc = compiler.CompileExpression<float, float, float, float>(content, {"x", "y", "z"}, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(3, 2, 7), std::fma(3, 2, 7));
}
