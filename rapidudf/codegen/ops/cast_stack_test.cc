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
#include "rapidudf/codegen/ops/cast.h"
#include "rapidudf/log/log.h"

#include "xbyak/xbyak_util.h"

using namespace rapidudf;
using namespace Xbyak::util;
template <typename F, typename T>
static std::pair<T, T> test_int_cast_int(F x, bool dump_asm = false) {
  CodeGenerator c(4096, false);

  auto scr_dtype = get_dtype<F>();
  auto var = c.NewValue(scr_dtype);
  auto arg = c.NewValueByRegister(scr_dtype, xmm0);
  auto src_addr = var->GetStackAddress();
  c.GetCodeGen().mov(var->GetStackAddress(), rdi.changeBit(src_addr.getBit()));

  auto dst_dtype = get_dtype<T>();
  auto ret_var = c.NewValue(dst_dtype);
  static_cast_value(c.GetCodeGen(), var->GetStackAddress(), scr_dtype, ret_var->GetStackAddress(), dst_dtype);
  ret_var->Mov(rax);
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<T, F>();
  return {static_cast<T>(x), f(x)};
}

template <typename F, typename T>
static std::pair<T, T> test_float_cast_int(F x, bool dump_asm = false) {
  CodeGenerator c(4096, false);
  auto scr_dtype = get_dtype<F>();
  auto var = c.NewValue(scr_dtype);
  auto arg = c.NewValueByRegister(scr_dtype, xmm0);
  if (scr_dtype == DATA_F32) {
    c.GetCodeGen().movd(var->GetStackAddress(), xmm0);
  } else {
    c.GetCodeGen().movq(var->GetStackAddress(), xmm0);
  }

  auto dst_dtype = get_dtype<T>();
  auto ret_var = c.NewValue(dst_dtype);
  static_cast_value(c.GetCodeGen(), var->GetStackAddress(), scr_dtype, ret_var->GetStackAddress(), dst_dtype);
  ret_var->Mov(rax);
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<T, F>();
  return {static_cast<T>(x), f(x)};
}
template <typename F, typename T>
static std::pair<T, T> test_any_cast_float(F x, bool dump_asm = false) {
  CodeGenerator c(4096, false);

  auto scr_dtype = get_dtype<F>();
  auto var = c.NewValue(scr_dtype);
  auto arg = c.NewValueByRegister(scr_dtype, xmm0);
  auto src_addr = var->GetStackAddress();
  if (scr_dtype == DATA_F32) {
    c.GetCodeGen().movd(src_addr, xmm0);
  } else if (scr_dtype == DATA_F64) {
    c.GetCodeGen().movq(src_addr, xmm0);
  } else {
    c.GetCodeGen().mov(src_addr, rdi.changeBit(src_addr.getBit()));
  }
  auto dst_dtype = get_dtype<T>();
  auto ret_var = c.NewValue(dst_dtype);
  static_cast_value(c.GetCodeGen(), var->GetStackAddress(), scr_dtype, ret_var->GetStackAddress(), dst_dtype);
  ret_var->Mov(xmm0);
  c.Finish();
  if (dump_asm) {
    c.DumpAsm();
  }
  auto f = c.GetFunc<T, F>();
  return {static_cast<T>(x), f(x)};
}

TEST(Cast, float_float) {
  constexpr int test_count = 3;
  float test_f[test_count] = {-1.1, 2.3, 0.5};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<float, double>(test_f[i]);
    ASSERT_EQ(a0, a1);
  }
  float test_df[test_count] = {-1.122, 2.333, 0.556};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<double, float>(test_df[i]);
    ASSERT_EQ(a0, a1);
  }
}

TEST(Cast, f32_ints) {
  constexpr int test_count = 3;
  float test_x[test_count] = {-1.1, 2.3, 0.5};

  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, int8_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, uint8_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, int32_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, uint32_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, int16_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, uint16_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, int64_t>(test_x[i], false);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<float, uint64_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
}

TEST(Cast, f64_ints) {
  constexpr int test_count = 3;
  double test_x[test_count] = {-0.1, 10000.3, 100.2345};

  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, int8_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, uint8_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, int32_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, uint32_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, int16_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, uint16_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, int64_t>(test_x[i], false);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_float_cast_int<double, uint64_t>(test_x[i]);
    ASSERT_EQ(a0, a1);
  }
}

TEST(Cast, int64_ints) {
  constexpr int test_count = 3;
  int64_t test_i64[test_count] = {-10, 10, 10002};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, int8_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, int16_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, int32_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, uint8_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, uint16_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, uint32_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int64_t, uint64_t>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
}
TEST(Cast, int32_ints) {
  constexpr int test_count = 3;
  int32_t test_i32[test_count] = {-10, 10, 10002};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, int8_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, int16_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, int64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, uint8_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, uint16_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, uint32_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int32_t, uint64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
}
TEST(Cast, int16_ints) {
  constexpr int test_count = 3;
  int16_t test_i32[test_count] = {-10, 10, 10002};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, int8_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, int32_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, int64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, uint8_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, uint16_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, uint32_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int16_t, uint64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
}

TEST(Cast, int8_ints) {
  constexpr int test_count = 3;
  int8_t test_i32[test_count] = {-10, 10, 127};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, int16_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, int32_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, int64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, uint8_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, uint16_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, uint32_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_int_cast_int<int8_t, uint64_t>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
}

TEST(Cast, int_floats) {
  constexpr int test_count = 3;
  int8_t test_i8[test_count] = {-10, 10, 127};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int8_t, float>(test_i8[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int8_t, double>(test_i8[i]);
    ASSERT_EQ(a0, a1);
  }

  int16_t test_i16[test_count] = {-10, 10, 12700};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int16_t, float>(test_i16[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int16_t, double>(test_i16[i]);
    ASSERT_EQ(a0, a1);
  }

  int32_t test_i32[test_count] = {-10, 10, 12700};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int32_t, float>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int32_t, double>(test_i32[i]);
    ASSERT_EQ(a0, a1);
  }

  uint32_t test_u32[test_count] = {100, 10, 12700};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<uint32_t, float>(test_u32[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<uint32_t, double>(test_u32[i]);
    ASSERT_EQ(a0, a1);
  }

  int32_t test_i64[test_count] = {-10, 10, 12700};
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int64_t, float>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
  for (int i = 0; i < test_count; i++) {
    auto [a0, a1] = test_any_cast_float<int64_t, double>(test_i64[i]);
    ASSERT_EQ(a0, a1);
  }
}