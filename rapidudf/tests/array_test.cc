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
#include <vector>
#include "absl/strings/str_join.h"

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/vector.h"

using namespace rapidudf;

static void print_span(absl::Span<const StringView> x) {
  RUDF_ERROR("@@@{}", x.size());
  for (auto v : x) {
    RUDF_ERROR("{}", v);
  }
}
RUDF_FUNC_REGISTER(print_span)
TEST(JitCompiler, array_simple) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  std::string source = R"(
     print_span(["ehllo", "adas", "aas"])
  )";
  auto result = compiler.CompileExpression<void>(source, {});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  f();

  std::string source1 = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x,simd_vector<f64> y){
        return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
      // return  x * y;
    }
  )";

  auto result1 =
      compiler.CompileFunction<simd_vector_f64, rapidudf::Context&, simd_vector_f64, simd_vector_f64>(source1);
  if (!result1.ok()) {
    RUDF_ERROR("###{}", result1.status().ToString());
  }
  auto ff = std::move(result1.value());
  std::vector<double> xx, yy, actuals, final_results;
  size_t test_n = 16;
  for (size_t i = 0; i < test_n; i++) {
    xx.emplace_back(i + 1);
    yy.emplace_back(i + 101);
  }
  rapidudf::Context ctx;
  auto fr = ff(ctx, ctx.NewVector(xx), ctx.NewVector(yy));
}

TEST(JitCompiler, vector_cmp) {
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      x>5&&x<5;
      return 5+x;
    }
  )";
  auto rc = compiler.CompileFunction<simd_vector_i32, Context&, simd_vector_i32>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, ctx.NewVector(vec));
  ASSERT_EQ(result->Size(), vec.size());
  for (size_t i = 0; i < result->Size(); i++) {
    ASSERT_FLOAT_EQ((*result)[i], vec[i] + 5);
  }
}