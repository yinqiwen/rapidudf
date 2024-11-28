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
#include <cmath>
#include <functional>
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, sqrt) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = R"(
    float test_func(float x){
      return sqrt(x);
    }
  )";
  auto rc = compiler.CompileFunction<float, float>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  float v = 1.2;
  ASSERT_FLOAT_EQ(f(v), sqrt(v));
}

TEST(JitCompiler, tanh) {
  JitCompiler compiler;
  std::string content = "tanh(x)";
  double x = 3.14;
  auto rc1 = compiler.CompileExpression<double, double>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_DOUBLE_EQ(f1(x), std::tanh(x));
}

TEST(JitCompiler, complex_math) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = R"(
    double test_func(double a, double t, double c){
      return a * exp(2.2/3.3*t) + c;
    }
  )";
  auto rc = compiler.CompileFunction<double, double, double, double>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  double a = 1.2;
  double t = 2.1;
  double c = 3.3;

  double v = a * std::exp(2.2 / 3.3 * t) + c;
  ASSERT_DOUBLE_EQ(f(a, t, c), v);
}

TEST(JitCompiler, random) {
  JitCompiler compiler;
  std::string content = "random(now_s())";

  auto rc1 = compiler.CompileExpression<uint64_t, uint64_t>(content, {"x"});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  RUDF_INFO("{}", f1(1));
  RUDF_INFO("{}", f1(2));
  RUDF_INFO("{}", f1(3));
}
