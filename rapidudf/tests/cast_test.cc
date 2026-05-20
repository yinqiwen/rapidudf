/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include "rapidudf/rapidudf.h"
using namespace rapidudf;

TEST(JitCompiler, u64_f64) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<double, uint64_t>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_DOUBLE_EQ(f(11), 11);
  ASSERT_DOUBLE_EQ(f(12121), 12121);
}

TEST(JitCompiler, f64_u64) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<uint64_t, double>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(11.1), 11);
  ASSERT_EQ(f(12121.2), 12121);
}

// Regression: signed integer extension must use sext, not zext.
// e.g. int8_t(-1) widened to int32_t should remain -1, not 255.
TEST(JitCompiler, i8_to_i32_signed_extension) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<int32_t, int8_t>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(static_cast<int8_t>(-1)), -1);
  ASSERT_EQ(f(static_cast<int8_t>(-128)), -128);
  ASSERT_EQ(f(static_cast<int8_t>(127)), 127);
}

TEST(JitCompiler, i16_to_i64_signed_extension) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<int64_t, int16_t>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(static_cast<int16_t>(-1)), -1LL);
  ASSERT_EQ(f(static_cast<int16_t>(-32768)), -32768LL);
}

TEST(JitCompiler, u8_to_u32_zero_extension) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = "x";
  auto rc = compiler.CompileExpression<uint32_t, uint8_t>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(static_cast<uint8_t>(255)), 255U);
  ASSERT_EQ(f(static_cast<uint8_t>(0)), 0U);
}

// Regression: scalar floating-point widening must use fpext (not fptrunc).
TEST(JitCompiler, f32_to_f64_widen) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<double, float>("x", {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_DOUBLE_EQ(f(1.5f), 1.5);
  ASSERT_DOUBLE_EQ(f(-3.25f), -3.25);
}

// Regression: scalar floating-point narrowing must use fptrunc (not fpext).
TEST(JitCompiler, f64_to_f32_narrow) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<float, double>("x", {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(2.5), 2.5f);
  ASSERT_FLOAT_EQ(f(-7.125), -7.125f);
}

// Regression: mixing an f32 variable with f64 literals should compile and
// run correctly. Previously this triggered "DestTy too big for FPTrunc"
// during JIT codegen because the FPExt/FPTrunc branches were swapped.
TEST(JitCompiler, f32_var_with_f64_literal) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<float, float>("x * 1.5 + 2.25", {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_FLOAT_EQ(f(2.0f), 2.0f * 1.5f + 2.25f);
  ASSERT_FLOAT_EQ(f(-4.0f), -4.0f * 1.5f + 2.25f);
}
