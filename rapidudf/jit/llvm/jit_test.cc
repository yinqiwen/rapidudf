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

#include "rapidudf/jit/llvm/jit.h"
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"

using namespace rapidudf;
using namespace rapidudf::llvm;

struct TestBase {
  int x;
  int a;
  float y = 1111.21;
  std::string str = "hello";
};
RUDF_STRUCT_FIELDS(TestBase, y);

struct TestA {
  int x;
  int a;
  TestBase base;
  std::string str = "hello";
};
RUDF_STRUCT_FIELDS(TestA, base, str);

static void test_extern_func() { RUDF_INFO("###test invoke!!!"); }
RUDF_FUNC_REGISTER(test_extern_func, rapidudf::kFuncNoAttrs)

// TEST(JitCompiler, simple) {
//   spdlog::set_level(spdlog::level::debug);
//   JitCompiler compiler;
//   std::string content = R"(
//     int test_func(int x, int y){
//       x-=y;
//       test_extern_func();
//       return x + 1.2;
//     }
//   )";

//   auto rc = compiler.CompileFunction<int, int, int>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("####{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   ASSERT_DOUBLE_EQ(f(2, 3), -1);
// }
// TEST(JitCompiler, string_view) {
//   spdlog::set_level(spdlog::level::debug);
//   JitCompiler compiler;
//   std::string content = R"(
//     bool test_func(TestA x){
//       var s = "hello";
//       return x.str == s;
//     }
//   )";

//   auto rc = compiler.CompileFunction<bool, const TestA&>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("####{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   TestA t;
//   auto f = std::move(rc.value());
//   ASSERT_EQ(f(t), true);
// }

// TEST(JitCompiler, field_access) {
//   spdlog::set_level(spdlog::level::debug);
//   JitCompiler compiler;
//   std::string content = R"(
//     f32 test_func(TestA x){
//       return x.base.y;
//     }
//   )";

//   auto rc = compiler.CompileFunction<float, const TestA&>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("####{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   TestA t;

//   ASSERT_FLOAT_EQ(f(t), t.base.y);
// }

// static void test_span_func(absl::Span<int8_t> span) { RUDF_INFO("###span size:{}", absl::StrJoin(span, ",")); }
// RUDF_FUNC_REGISTER(test_span_func, rapidudf::kFuncNoAttrs)

// TEST(JitCompiler, span) {
//   spdlog::set_level(spdlog::level::debug);
//   JitCompiler compiler;
//   std::string content = R"(
//     void test_func(){
//       test_span_func([1_i8,2,3]);
//       return;
//     }
//   )";

//   auto rc = compiler.CompileFunction<void>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("####{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   TestA t;
//   auto f = std::move(rc.value());
//   f();
// }

// TEST(JitCompiler, ternary) {
//   spdlog::set_level(spdlog::level::debug);
//   JitCompiler compiler;
//   std::string content = R"(
//     int test_func(int x){
//       return x>3?1:0;
//     }
//   )";

//   auto rc = compiler.CompileFunction<int, int>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("####{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   TestA t;
//   auto f = std::move(rc.value());
//   ASSERT_EQ(f(2), 0);
//   ASSERT_EQ(f(5), 1);
// }
