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
