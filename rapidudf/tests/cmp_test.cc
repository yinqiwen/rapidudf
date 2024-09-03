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
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, cmp_greater) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(int x, int y){ 
      return x>y;
    }
  )";
  auto rc = compiler.CompileFunction<int, int, int>(content);
  // ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int, int, int>(true);
  // ASSERT_TRUE(f != nullptr);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 1);
  ASSERT_EQ(f(1, 10), 0);
  ASSERT_EQ(f(1, 1), 0);
}

TEST(JitCompiler, cmp_greater_eq) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(int x, int y){ 
      return x>=y;
    }
  )";
  auto rc = compiler.CompileFunction<bool, int, int>(content);
  // ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, int, int>(true);
  // ASSERT_TRUE(f != nullptr);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 1);
  ASSERT_EQ(f(1, 10), 0);
  ASSERT_EQ(f(1, 1), 1);
}

TEST(JitCompiler, cmp_less) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(float x, float y){ 
      return x<y;
    }
  )";
  auto rc = compiler.CompileFunction<int, float, float>(content);
  // ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int, float, float>(true);
  // ASSERT_TRUE(f != nullptr);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 0);
  ASSERT_EQ(f(1, 10), 1);
  ASSERT_EQ(f(1, 1), 0);
}

TEST(JitCompiler, cmp_less_eq) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(double x, double y){
      return x<=y;
    }
  )";
  auto rc = compiler.CompileFunction<bool, double, double>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, double, double>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 0);
  ASSERT_EQ(f(1, 10), 1);
  ASSERT_EQ(f(1, 1), 1);
}

TEST(JitCompiler, cmp_eq) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(float x, float y){ 
      return x==y;
    }
  )";
  auto rc = compiler.CompileFunction<int, float, float>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int, float, float>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 0);
  ASSERT_EQ(f(1, 10), 0);
  ASSERT_EQ(f(1, 1), 1);
}

TEST(JitCompiler, cmp_neq) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(double x, double y){
      return x!=y;
    }
  )";
  auto rc = compiler.CompileFunction<bool, double, double>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, double, double>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 1);
  ASSERT_EQ(f(1, 10), 1);
  ASSERT_EQ(f(1, 1), 0);
}

TEST(JitCompiler, cmp_string) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    bool test_func(string_view x, string_view y){
      return x>y;
    }
  )";
  auto rc = compiler.CompileFunction<bool, StringView, StringView>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<bool, std::string_view, std::string_view>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_FALSE(f("h", "h"));
  ASSERT_TRUE(f("hello", "h"));
  ASSERT_FALSE(f("h", "hello"));
}