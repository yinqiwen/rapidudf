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
TEST(JitCompiler, u64_f64) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<double, uint64_t>(content, {"x"}, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_DOUBLE_EQ(f(11), 11);
  ASSERT_DOUBLE_EQ(f(12121), 12121);
}

TEST(JitCompiler, f64_u64) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<uint64_t, double>(content, {"x"}, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(11.1), 11);
  ASSERT_EQ(f(12121.2), 12121);
}
