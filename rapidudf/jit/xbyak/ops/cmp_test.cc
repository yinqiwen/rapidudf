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

#include "rapidudf/jit/xbyak/code_generator.h"
#include "rapidudf/jit/xbyak/ops/cmp.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

#include "xbyak/xbyak_util.h"

using namespace rapidudf;
using namespace Xbyak::util;
template <typename T>
static int std_cmp(T left, T right) {
  if (left > right) {
    return 1;
  }

  if (left == right) {
    return 0;
  }

  return -1;
}

template <typename T, bool reverse = false>
std::pair<int, int> test_cmp_reg_const(T x, T y, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto dtype = get_dtype<T>();
  uint32_t bits = dtype.Bits();
  // LSE_INFO("#######src_bits:{}", src_bits);
  auto const_val = c.NewConstValue(dtype);
  const_val->Set(y);
  if (dtype == DATA_F32 || dtype == DATA_F64) {
    cmp_value(c.GetCodeGen(), dtype, xmm0, const_val->GetConstBin(), reverse);
  } else {
    cmp_value(c.GetCodeGen(), dtype, rdi.changeBit(bits), const_val->GetConstBin(), reverse);
  }
  c.Jump("eq", OP_EQUAL);
  c.Jump("gt", OP_GREATER);
  c.Jump("lt", OP_LESS);

  c.Label("gt");
  c.GetCodeGen().mov(eax, 1);
  c.Jump("exit");

  c.Jump("exit");
  c.Label("eq");
  c.GetCodeGen().mov(eax, 0);
  c.Jump("exit");
  c.Label("lt");
  c.GetCodeGen().mov(eax, -1);
  c.Jump("exit");
  c.Label("exit");
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<int, T>();
  int actual = std_cmp(x, y);
  if (reverse) {
    actual = std_cmp(y, x);
  }
  return {actual, f(x)};
}

TEST(Cmp, register_cmp_reg_const) {
  constexpr int test_count = 3;
  int64_t test_i640[test_count] = {-10, 10, 102};
  int64_t test_i641[test_count] = {-100, 10, 101};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<int64_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<int64_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<int32_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<int32_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<int16_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<int16_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<int8_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<int8_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }

  double test_f640[test_count] = {-10.1, 10.123, 102.456};
  double test_f641[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<double>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<double, true>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a2, a3);
  }

  float test_f320[test_count] = {-10.1, 10.123, 102.456};
  float test_f321[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_const<float>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_const<float, true>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a2, a3);
  }
}

template <typename T, bool use_xmm = false>
std::pair<int, int> test_cmp_reg_reg(T x, T y, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto dtype = get_dtype<T>();
  uint32_t bits = dtype.Bits();
  auto val = c.NewValueByRegister(dtype, r10);
  if (use_xmm) {
    val = c.NewValueByRegister(dtype, xmm15);
  }
  val->Set(y);
  if (dtype == DATA_F32 || dtype == DATA_F64) {
    cmp_value(c.GetCodeGen(), dtype, xmm0, val->GetRegister());
  } else {
    cmp_value(c.GetCodeGen(), dtype, rdi.changeBit(bits), val->GetRegister());
  }
  c.Jump("eq", OP_EQUAL);
  c.Jump("gt", OP_GREATER);
  c.Jump("lt", OP_LESS);

  c.Label("gt");
  c.GetCodeGen().mov(eax, 1);
  c.Jump("exit");

  c.Jump("exit");
  c.Label("eq");
  c.GetCodeGen().mov(eax, 0);
  c.Jump("exit");
  c.Label("lt");
  c.GetCodeGen().mov(eax, -1);
  c.Jump("exit");
  c.Label("exit");
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<int, T>();
  int actual = std_cmp(x, y);
  return {actual, f(x)};
}

TEST(Cmp, register_cmp_reg_reg) {
  constexpr int test_count = 3;
  int64_t test_i640[test_count] = {-10, 10, 102};
  int64_t test_i641[test_count] = {-100, 10, 101};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<int64_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<int64_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<int32_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<int32_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<int16_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<int16_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<int8_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<int8_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }

  double test_f640[test_count] = {-10.1, 10.123, 102.456};
  double test_f641[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<double>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<double, true>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a2, a3);
  }

  float test_f320[test_count] = {-10.1, 10.123, 102.456};
  float test_f321[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_reg<float>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_reg<float, true>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a2, a3);
  }
}

template <typename T, bool reverse = false>
std::pair<int, int> test_cmp_reg_stack(T x, T y, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto dtype = get_dtype<T>();
  uint32_t bits = dtype.Bits();
  auto val = c.NewValue(dtype);
  val->Set(y);
  if (dtype == DATA_F32 || dtype == DATA_F64) {
    cmp_value(c.GetCodeGen(), dtype, xmm0, val->GetStackAddress(), reverse);
  } else {
    cmp_value(c.GetCodeGen(), dtype, rdi.changeBit(bits), val->GetStackAddress(), reverse);
  }
  c.Jump("eq", OP_EQUAL);
  c.Jump("gt", OP_GREATER);
  c.Jump("lt", OP_LESS);

  c.Label("gt");
  c.GetCodeGen().mov(eax, 1);
  c.Jump("exit");

  c.Jump("exit");
  c.Label("eq");
  c.GetCodeGen().mov(eax, 0);
  c.Jump("exit");
  c.Label("lt");
  c.GetCodeGen().mov(eax, -1);
  c.Jump("exit");
  c.Label("exit");
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<int, T>();
  int actual = std_cmp(x, y);
  if (reverse) {
    actual = std_cmp(y, x);
  }
  return {actual, f(x)};
}

TEST(Cmp, register_cmp_reg_stack) {
  constexpr int test_count = 3;
  int64_t test_i640[test_count] = {-10, 10, 102};
  int64_t test_i641[test_count] = {-100, 10, 101};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<int64_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<int64_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<int32_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<int32_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<int16_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<int16_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<int8_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<int8_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }

  double test_f640[test_count] = {-10.1, 10.123, 102.456};
  double test_f641[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<double>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<double, true>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a2, a3);
  }

  float test_f320[test_count] = {-10.1, 10.123, 102.456};
  float test_f321[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_reg_stack<float>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_reg_stack<float, true>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a2, a3);
  }
}

template <typename T>
std::pair<int, int> test_cmp_stack_stack(T x, T y, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto dtype = get_dtype<T>();
  auto left = c.NewValue(dtype);
  left->Set(x);
  auto right = c.NewValue(dtype);
  right->Set(y);
  cmp_value(c.GetCodeGen(), dtype, left->GetStackAddress(), right->GetStackAddress());
  c.Jump("eq", OP_EQUAL);
  c.Jump("gt", OP_GREATER);
  c.Jump("lt", OP_LESS);

  c.Label("gt");
  c.GetCodeGen().mov(eax, 1);
  c.Jump("exit");

  c.Jump("exit");
  c.Label("eq");
  c.GetCodeGen().mov(eax, 0);
  c.Jump("exit");
  c.Label("lt");
  c.GetCodeGen().mov(eax, -1);
  c.Jump("exit");
  c.Label("exit");
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<int>();
  int actual = std_cmp(x, y);
  return {actual, f()};
}

TEST(Cmp, register_cmp_stack_stack) {
  constexpr int test_count = 3;
  int64_t test_i640[test_count] = {-10, 10, 102};
  int64_t test_i641[test_count] = {-100, 10, 101};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<int64_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<int32_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<int16_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<int8_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
  }

  double test_f640[test_count] = {-10.1, 10.123, 102.456};
  double test_f641[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<double>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a0, a1);
  }

  float test_f320[test_count] = {-10.1, 10.123, 102.456};
  float test_f321[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_stack<float>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a0, a1);
  }
}

template <typename T, bool reverse = false>
std::pair<int, int> test_cmp_stack_const(T x, T y, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto dtype = get_dtype<T>();

  auto left = c.NewValue(dtype);
  left->Set(x);
  auto right = c.NewConstValue(dtype);
  right->Set(y);
  cmp_value(c.GetCodeGen(), dtype, left->GetStackAddress(), right->GetConstBin(), reverse);

  c.Jump("eq", OP_EQUAL);
  c.Jump("gt", OP_GREATER);
  c.Jump("lt", OP_LESS);

  c.Label("gt");
  c.GetCodeGen().mov(eax, 1);
  c.Jump("exit");

  c.Jump("exit");
  c.Label("eq");
  c.GetCodeGen().mov(eax, 0);
  c.Jump("exit");
  c.Label("lt");
  c.GetCodeGen().mov(eax, -1);
  c.Jump("exit");
  c.Label("exit");
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<int>();
  int actual = std_cmp(x, y);
  if (reverse) {
    actual = std_cmp(y, x);
  }
  return {actual, f()};
}

TEST(Cmp, register_cmp_stack_const) {
  constexpr int test_count = 3;
  int64_t test_i640[test_count] = {-10, 10, 102};
  int64_t test_i641[test_count] = {-100, 10, 101};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<int64_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<int64_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<int32_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<int32_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<int16_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<int16_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<int8_t>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<int8_t, true>(test_i640[i], test_i641[i], false);
    ASSERT_EQ(a2, a3);
  }

  double test_f640[test_count] = {-10.1, 10.123, 102.456};
  double test_f641[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<double>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<double, true>(test_f640[i], test_f641[i], false);
    ASSERT_EQ(a2, a3);
  }

  float test_f320[test_count] = {-10.1, 10.123, 102.456};
  float test_f321[test_count] = {-100.23, 10.123, 101.345};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_cmp_stack_const<float>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a0, a1);
    auto [a2, a3] = test_cmp_stack_const<float, true>(test_f320[i], test_f321[i], false);
    ASSERT_EQ(a2, a3);
  }

  RUDF_INFO("rax:{}, xmm0:{}, rsi:{}, eax:{}", rax.getIdx(), xmm0.getIdx(), rsi.getIdx(), eax.getIdx());
}