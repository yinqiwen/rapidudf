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

#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"

struct TestStruct {};

using namespace rapidudf;
using namespace Xbyak::util;

int test_func(int a, std::string_view b, float c) { return 0; }

TEST(Function, simple) {
  //   std::function<int(int, std::string_view, float)> f = test_func;
  FuncRegister reg("name", test_func);
  RUDF_FUNC_REGISTER(test_func);

  auto ts = get_dtypes<std::tuple<int, float, int64_t>>();
  ASSERT_EQ(ts.size(), 3);
  ASSERT_EQ(ts[0], DATA_I32);
  ASSERT_EQ(ts[1], DATA_F32);

  ts = get_dtypes<std::string>();
  ASSERT_EQ(ts.size(), 1);
  ASSERT_EQ(ts[0], DATA_STRING);

  ts = get_dtypes<std::pair<int, float>>();
  ASSERT_EQ(ts.size(), 2);
  ASSERT_EQ(ts[0], DATA_I32);
  ASSERT_EQ(ts[1], DATA_F32);
}

struct T0 {
  double d;
  float f;
};

std::pair<float, float> test_pair0() {}
std::pair<int64_t, int32_t> test_pair1() {}
std::pair<float, int> test_pair2() {}
std::pair<int, float> test_pair3() {}
std::tuple<float, float, float, int> test_pair4() {}
std::tuple<int, std::string_view, float> test_pair5() {}
TEST(Function, ret_pair) {
  RUDF_FUNC_REGISTER(test_pair0);
  RUDF_FUNC_REGISTER(test_pair1);
  RUDF_FUNC_REGISTER(test_pair2);
  RUDF_FUNC_REGISTER(test_pair3);
  RUDF_FUNC_REGISTER(test_pair4);
  RUDF_FUNC_REGISTER(test_pair5);
  auto desc = FuncFactory::GetFunc("test_pair0");
  uint32_t total_bits = 0;
  auto regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(total_bits, 64);
  ASSERT_EQ(regs.size(), 1);

  ASSERT_TRUE(regs[0]->isXMM() && regs[0]->getIdx() == 0);

  desc = FuncFactory::GetFunc("test_pair1");
  regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(total_bits, sizeof(std::pair<int64_t, int32_t>) * 8);
  ASSERT_EQ(regs.size(), 2);
  ASSERT_TRUE(regs[0]->isREG() && regs[0]->getIdx() == rax.getIdx());
  ASSERT_TRUE(regs[1]->isREG() && regs[1]->getIdx() == rdx.getIdx());

  desc = FuncFactory::GetFunc("test_pair2");
  regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(total_bits, sizeof(std::pair<float, int>) * 8);
  ASSERT_EQ(regs.size(), 1);
  ASSERT_TRUE(regs[0]->isREG() && regs[0]->getIdx() == rax.getIdx());

  desc = FuncFactory::GetFunc("test_pair3");
  regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(total_bits, sizeof(std::pair<int, float>) * 8);
  ASSERT_EQ(regs.size(), 1);
  ASSERT_TRUE(regs[0]->isREG() && regs[0]->getIdx() == rax.getIdx());

  desc = FuncFactory::GetFunc("test_pair4");
  regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(total_bits, sizeof(std::tuple<float, float, float, int>) * 8);
  ASSERT_EQ(regs.size(), 2);
  ASSERT_TRUE(regs[0]->isXMM() && regs[0]->getIdx() == xmm0.getIdx());
  ASSERT_TRUE(regs[1]->isREG() && regs[1]->getIdx() == rax.getIdx());

  desc = FuncFactory::GetFunc("test_pair5");
  regs = desc->GetReturnValueRegisters(total_bits);
  ASSERT_EQ(regs.size(), 0);
  ASSERT_EQ(total_bits, sizeof(std::tuple<int, std::string_view, float>) * 8);
}