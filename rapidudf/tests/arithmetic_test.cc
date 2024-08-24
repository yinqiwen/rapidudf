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
TEST(JitCompiler, add) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    int test_func(int x, int y){
      return x+y;
    }
  )";
  // auto func = parse_function_ast(ctx, content);
  // ASSERT_TRUE(func.has_value());
  auto rc = compiler.CompileFunction<int, int, int>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int, int, int>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(1, 2), 3);
  ASSERT_EQ(f(111, 222), 333);
}
TEST(JitCompiler, sub) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    float test_func(float x, float y){
      return x-y;
    }
  )";
  auto rc = compiler.CompileFunction<float, float, float>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<float, float, float>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(3.1, 1.2), 1.9);
  ASSERT_FLOAT_EQ(f(5.9, 1.2), 4.7);
}

TEST(JitCompiler, multiply) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    u64 test_func(u64 x, u64 y){
      return x*y;
    }
  )";
  auto rc = compiler.CompileFunction<uint64_t, uint64_t, uint64_t>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<uint64_t, uint64_t, uint64_t>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(10, 20), 200);
  ASSERT_FLOAT_EQ(f(1, 111), 111);
}

TEST(JitCompiler, divid) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    u64 test_func(u64 x, u64 y){
      return x/y;
    }
  )";
  auto rc = compiler.CompileFunction<uint64_t, uint64_t, uint64_t>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<uint64_t, uint64_t, uint64_t>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(100, 20), 5);
  ASSERT_FLOAT_EQ(f(1, 111), 0);
}

TEST(JitCompiler, mod) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    u64 test_func(u64 x, u64 y){
      return x%y;
    }
  )";
  auto rc = compiler.CompileFunction<uint64_t, uint64_t, uint64_t>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<uint64_t, uint64_t, uint64_t>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(100, 20), 0);
  ASSERT_FLOAT_EQ(f(7, 5), 2);
}