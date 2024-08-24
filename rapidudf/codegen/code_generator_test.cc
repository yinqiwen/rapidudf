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

#include "rapidudf/codegen/code_generator.h"
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/log/log.h"

#include "xbyak/xbyak_util.h"

struct TestStruct {};

using namespace rapidudf;

using namespace Xbyak::util;

void print_rsp(uint64_t v) {
  //
  RUDF_INFO("RSP: {} ", v);
}
void noop() { int a = 0; }

int test_func0(int a, double b) {
  RUDF_INFO("invoke test_func:{} {}  ", a, b);
  return a + 10;
}
int test_func(int a, double b, short c, bool x) {
  RUDF_INFO("invoke test_func:{} {} {} {} ", a, b, c, x);
  return a + 10;
}
RUDF_FUNC_REGISTER(noop);
RUDF_FUNC_REGISTER(test_func0);
RUDF_FUNC_REGISTER(test_func);
RUDF_FUNC_REGISTER(print_rsp);

// TEST(codegen, simple) {
//   CodeGenerator c(40960, true);
//   // c.GetCodeGen().push(r15);
//   // c.CallFunction("noop", {});
//   c.CallFunction("print_rsp", {c.NewValueByRegister(DATA_U64, rsp)});
//   // c.GetCodeGen().pop(r15);
//   c.Finish();
//   c.DumpAsm();

//   auto f = c.GetFunc<void>();
//   f();
// }

// TEST(codegen, func_call0) {
//   CodeGenerator c(40960, true);
//   // c.CallFunction("print_rsp", {c.NewConstValue(DATA_U64, 111)});
//   c.CallFunction("print_rsp", {c.NewValueByRegister(DATA_U64, rsp)});
//   c.Finish();
//   c.DumpAsm();

//   auto f = c.GetFunc<void>();
//   f();
// }
// TEST(codegen, func_call1) {
//   CodeGenerator codegen(4096, true);
//   std::vector<ValuePtr> args;
//   auto a = codegen.NewConstValue(DATA_I32, 101);
//   auto b = codegen.NewConstValue(DATA_F64);
//   b->Set(2.79);
//   auto c = codegen.NewConstValue(DATA_I32, 103);
//   auto x = codegen.NewConstValue(DATA_U8, 1);
//   args.push_back(a);
//   args.push_back(b);
//   args.push_back(c);
//   args.push_back(x);
//   codegen.CallFunction("test_func", args);
//   codegen.CallFunction("print_rsp", {codegen.NewValueByRegister(DATA_U64, rsp)});
//   codegen.Finish();
//   codegen.DumpAsm();
//   auto f = codegen.GetFunc<void>();
//   f();
// }
TEST(codegen, func_call2) {
  spdlog::set_level(spdlog::level::debug);
  CodeGenerator codegen(4096, true);
  std::vector<ValuePtr> args;
  auto a = codegen.NewValue(DATA_I32);
  a->Set(101);
  auto b = codegen.NewValue(DATA_F64);
  b->Set(2.79);
  auto c = codegen.NewValue(DATA_I32);
  c->Set(202);
  auto x = codegen.NewValue(DATA_U8, 1);
  x->Set(0);
  args.push_back(a);
  args.push_back(b);
  args.push_back(c);
  args.push_back(x);
  codegen.CallFunction("test_func", args);
  codegen.Finish();
  codegen.DumpAsm();
  auto f = codegen.GetFunc<void>();
  f();
}

TEST(codegen, complex) {
  CodeGenerator c(4096, true);

  c.Jump("print_rsp", OP_EQUAL);
  // c.test();
  auto a = c.NewConstValue(DATA_I32);
  a->Set(101);
  auto b = c.NewConstValue(DATA_F64);
  b->Set(2.79);
  auto c_ = c.NewConstValue(DATA_I32);
  c_->Set(103);
  auto x = c.NewConstValue(DATA_U8);
  x->Set(1);
  std::vector<ValuePtr> args;
  args.push_back(a);
  args.push_back(b);
  args.push_back(c_);
  args.push_back(x);
  c.CallFunction("test_func", args);
  c_.reset();
  a.reset();
  b.reset();
  x.reset();
  args.clear();

  c.CallFunction("print_rsp", {c.NewValueByRegister(DATA_U64, rsp)});

  a = c.NewValue(DATA_I32);
  a->Set(102);
  b = c.NewValue(DATA_F64);
  b->Set(2.89);
  c_ = c.NewValue(DATA_I32);
  c_->Set(104);
  x = c.NewValue(DATA_U8);
  x->Set(0);
  args.push_back(a);
  args.push_back(b);
  args.push_back(c_);
  args.push_back(x);
  c.CallFunction("test_func", args);
  c_.reset();
  a.reset();
  b.reset();
  x.reset();
  args.clear();
  // c.CallFunction("print_rsp", {c.NewValueByRegister(DATA_U64, rsp)});

  // a = c.NewValue(DATA_I32, true);
  // RUDF_INFO("a stack:{}, reg:{}", a->IsStack(), a->IsRegister());
  // a->Set(7);
  // b = c.NewValue(DATA_F64);
  // b->Set(2.99);
  // c_ = c.NewValue(DATA_I64);
  // RUDF_INFO("c stack:{}, reg:{}", c_->IsStack(), c_->IsRegister());
  // c_->Set(9);
  // x = c.NewValue(DATA_U8, 0);
  // args.push_back(a);
  // args.push_back(b);
  // args.push_back(c_);
  // args.push_back(x);
  // c.CallFunction("test_func", args);
  // c_.reset();
  // a.reset();
  // b.reset();
  // x.reset();
  // args.clear();

  c.Label("print_rsp");
  c.CallFunction("print_rsp", {c.NewValueByRegister(DATA_U64, rsp)});
  c.Label("exit");
  c.Finish();
  c.DumpAsm();

  auto f = c.GetFunc<void>();

  f();

  // auto f = c.GetFunc<int, int>();

  // int n = f(200);
  // printf("###%d\n", n);
}