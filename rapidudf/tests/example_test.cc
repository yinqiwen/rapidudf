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
#include <locale>
#include <stdexcept>
#include <vector>
#include "rapidudf/codegen/function.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/reflect/macros.h"

struct TestMethodStruct {
  int a;
  float b;
  int* p = nullptr;
  std::string_view c;
  int get_a() const { return a; }
  void set_a(int x) { this->a = x; }
};
RUDF_STRUCT_MEMBER_METHODS(TestMethodStruct, get_a, set_a);

TEST(JitCompiler, member_methods) {
  spdlog::set_level(spdlog::level::debug);
  TestMethodStruct test;
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    int test_func(TestMethodStruct x, int a){
      x.set_a(a);
      return x.get_a();
    }
  )";
  auto rc = compiler.CompileFunction<int, TestMethodStruct&, int>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(test, 101), 101);
  ASSERT_EQ(test.a, 101);
}

struct MapHelper {
  static int map_get(const std::map<int, int>* map, int k) {
    auto found = map->find(k);
    if (found != map->end()) {
      return found->second;
    }
    return 0;
  }
};
RUDF_STRUCT_HELPER_METHODS_BIND(MapHelper, map_get)
TEST(JitCompiler, bind_member_methods) {
  spdlog::set_level(spdlog::level::debug);
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    int test_func(map<i32,i32> x, int a){
      return x.map_get(a);
    }
  )";
  auto rc = compiler.CompileFunction<int, const std::map<int, int>&, int>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  std::map<int, int> map = {{1, 1}, {2, 2}};
  ASSERT_EQ(f(map, 1), 1);
  ASSERT_EQ(f(map, 2), 2);
  ASSERT_EQ(f(map, 3), 0);
}

static int test_cpp_func(int x, int y) { return x * 100 + y; }
RUDF_FUNC_REGISTER(test_cpp_func, rapidudf::kFuncNoAttrs)

TEST(JitCompiler, cpp_func) {
  spdlog::set_level(spdlog::level::debug);
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    int test_func(int x, int y){
      return test_cpp_func(x, y);
    }
  )";
  auto rc = compiler.CompileFunction<int, int, int>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  std::map<int, int> map = {{1, 1}, {2, 2}};
  ASSERT_EQ(f(1, 2), 102);
  ASSERT_EQ(f(3, 2), 302);
}

struct ExceptionStruct {
  void test() { throw std::logic_error("test"); }
};
RUDF_STRUCT_SAFE_MEMBER_METHODS(ExceptionStruct, test);
TEST(JitCompiler, safe_member_func) {
  spdlog::set_level(spdlog::level::debug);
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    void test_func(ExceptionStruct t){
      t.test();
    }
  )";
  auto rc = compiler.CompileFunction<void, ExceptionStruct&>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ExceptionStruct test;
  ASSERT_ANY_THROW(f(test));
}

static void exception_func() { throw std::logic_error("exception_func"); }
RUDF_SAFE_FUNC_REGISTER(exception_func, rapidudf::kFuncNoAttrs)
TEST(JitCompiler, safe_func) {
  spdlog::set_level(spdlog::level::debug);
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    void test_func(){
      exception_func();
    }
  )";
  auto rc = compiler.CompileFunction<void>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_ANY_THROW(f());
}

TEST(JitCompiler, var) {
  spdlog::set_level(spdlog::level::debug);
  rapidudf::JitCompiler compiler;
  std::string source = R"(
    int test_func(int x, int y){
      var a = x+1;
      var b = y-1;
      return a + b;
    }
  )";
  auto result = compiler.CompileFunction<int, int, int>(source);
  auto f = std::move(result.value());
  int v = f(1, 1);
  ASSERT_EQ(v, 2);
}