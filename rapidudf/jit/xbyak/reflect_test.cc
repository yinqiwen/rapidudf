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
#include <vector>

#include "rapidudf/jit/xbyak/code_generator.h"
#include "rapidudf/jit/xbyak/ops/cast.h"
#include "rapidudf/jit/xbyak/ops/copy.h"
#include "rapidudf/jit/xbyak/reflect.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/reflect/struct.h"

#include "xbyak/xbyak_util.h"

using namespace Xbyak::util;
using namespace rapidudf;

struct TestBase {
  float y = 2.79;
};
RUDF_STRUCT_FIELDS(TestBase, y);
struct TestA {
  int x = 0;
  TestBase base;
  std::vector<TestBase> base_vec;
};
RUDF_STRUCT_FIELDS(TestA, x, base, base_vec);

struct TestStruct {
  int a;
  float b;
  int* p = nullptr;
  std::string_view c;
  TestA other;
};

// // template <typename T, typename RET, typename... Args>
// // static void test_f(RET (T::*f)(Args...)) {
// //   auto return_type = rapidudf::get_dtype<RET>();
// //   std::vector<rapidudf::DType> arg_types;
// //   (arg_types.emplace_back(rapidudf::get_dtype<Args>()), ...);
// // }

// // static void test_reflect() {
// //   auto* p = (TestStruct*)0;
// //   test_f(&TestStruct::get_a);
// //   auto x = &TestStruct::get_a;
// //   (p->*x)();
// // }

RUDF_STRUCT_FIELDS(TestStruct, a, b, c, p, other);

TEST(StructAccess, simple_int) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct x;
  x.a = 101;
  x.b = 3.14;

  CodeGenerator c(4096, false);
  auto builder = StructAccess<TestStruct>::GetStructMember("a");
  c.GetCodeGen().mov(rcx, rdi);
  DType dtype;
  auto result = BuildStructFieldAccess(c, *builder);
  ASSERT_TRUE(result.ok());
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<int, TestStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(x.a, aa);
}
TEST(StructAccess, simple_float) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct x;
  x.a = 101;
  x.b = 3.14;

  CodeGenerator c(4096, false);
  c.GetCodeGen().mov(rcx, rdi);
  auto builder = StructAccess<TestStruct>::GetStructMember("b");
  DType dtype;
  auto result = BuildStructFieldAccess(c, *builder);
  ASSERT_TRUE(result.ok());
  copy_value(c.GetCodeGen(), DATA_F32, eax, xmm0);
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<float, TestStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(x.b, aa);
}

TEST(StructAccess, simple_ptr) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct x;
  x.a = 101;
  x.b = 3.14;
  x.p = new int;

  CodeGenerator c(4096, false);
  c.GetCodeGen().mov(rcx, rdi);
  auto builder = StructAccess<TestStruct>::GetStructMember("p");
  DType dtype;
  auto result = BuildStructFieldAccess(c, *builder);
  ASSERT_TRUE(result.ok());
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<int*, TestStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(x.p, aa);
}

TEST(StructAccess, simple_string_view) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct x;
  x.a = 101;
  x.b = 3.14;
  x.p = new int;
  x.c = "hello,world";

  CodeGenerator c(4096, false);
  c.GetCodeGen().mov(rcx, rdi);
  auto builder = StructAccess<TestStruct>::GetStructMember("c");
  DType dtype;
  auto result = BuildStructFieldAccess(c, *builder);
  ASSERT_TRUE(result.ok());
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<std::string_view, TestStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(x.c, aa);
}

TEST(StructAccess, simple_other) {
  spdlog::set_level(spdlog::level::debug);
  TestStruct x;
  x.a = 101;
  x.b = 3.14;
  x.p = new int;
  x.c = "hello,world";

  CodeGenerator c(4096, false);
  c.GetCodeGen().mov(rcx, rdi);
  auto builder = StructAccess<TestStruct>::GetStructMember("other");
  DType dtype;
  auto result = BuildStructFieldAccess(c, *builder);
  ASSERT_TRUE(result.ok());
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<TestA*, TestStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(&x.other, aa);
}

struct TestMethodStruct {
  int a;
  float b;
  int* p = nullptr;
  std::string_view c;
  int get_a() const {
    RUDF_INFO("Called get_a");
    return a;
  }
  void set_a(int x) {
    RUDF_INFO("Called set_a");
    this->a = x;
  }
};
RUDF_STRUCT_SAFE_MEMBER_METHODS(TestMethodStruct, get_a, set_a);

TEST(StructAccess, get_method) {
  spdlog::set_level(spdlog::level::debug);
  TestMethodStruct x;
  x.a = 1011;

  CodeGenerator c(4096, false);
  auto this_val = c.NewValue(DATA_U64);
  this_val->Copy(&rdi);
  auto builder = StructAccess<TestMethodStruct>::GetStructMember("get_a");
  auto result = BuildStructFuncCall(c, *builder, *this_val, {});
  ASSERT_TRUE(result.ok());

  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<int, TestMethodStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(x.a, aa);
}
TEST(StructAccess, set_method) {
  spdlog::set_level(spdlog::level::debug);
  TestMethodStruct x;
  x.a = 1011;

  CodeGenerator c(4096, true);
  auto this_val = c.NewValue(DATA_U64);
  this_val->Copy(&rdi);

  auto set_builder = StructAccess<TestMethodStruct>::GetStructMember("set_a");
  auto val = c.NewConstValue(DATA_I32, 3333);
  auto result = BuildStructFuncCall(c, *set_builder, *this_val, {val});
  ASSERT_TRUE(result.ok());

  auto get_builder = StructAccess<TestMethodStruct>::GetStructMember("get_a");
  result = BuildStructFuncCall(c, *get_builder, *this_val, {});
  ASSERT_TRUE(result.ok());
  c.Finish();
  c.DumpAsm();
  auto f = c.GetFunc<int, TestMethodStruct*>();
  auto aa = f(&x);
  ASSERT_EQ(3333, aa);
}
