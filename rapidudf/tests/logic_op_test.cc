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
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"

using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, logic_and) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(int x, int y){
      return x==1&&y==2;
    }
  )";
  auto rc = compiler.CompileFunction<bool, int, int>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, int, int>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(1, 2));
  ASSERT_FALSE(f(1, 1));
  ASSERT_FALSE(f(2, 1));
}

TEST(JitCompiler, logic_or) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(int x, int y){
      return x==1||y==2;
    }
  )";
  auto rc = compiler.CompileFunction<bool, int, int>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, int, int>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(1, 2));
  ASSERT_TRUE(f(1, 1));
  ASSERT_TRUE(f(0, 2));
  ASSERT_FALSE(f(0, 0));
  ASSERT_FALSE(f(2, 1));
}
