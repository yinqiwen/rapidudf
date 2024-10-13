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