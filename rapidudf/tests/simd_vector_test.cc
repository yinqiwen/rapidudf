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

// SIMD-vector-specific structural tests: comparison, ternary, string compare,
// filter / gather / find search ops, custom user vector functions, and the
// distance-function library APIs.
//
// Pure math operator/function correctness lives in math_test.cc.

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

static float l2_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vectors must have the same length.");
  }

  float sum_of_squares = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0f, std::plus<>(),
                                            [](float a, float b) { return (a - b) * (a - b); });

  return std::sqrt(sum_of_squares);
}
static float dot_product(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vectors must have the same length.");
  }
  return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0f);
}

static float l2_norm(const std::vector<float>& vec) {
  float sum_of_squares = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f, std::plus<>(),
                                            [](float a, float b) { return a * b; });
  return std::sqrt(sum_of_squares);
}

static float cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Vectors must have the same length.");
  }

  float dot = dot_product(vec1, vec2);

  float norm1 = l2_norm(vec1);
  float norm2 = l2_norm(vec2);

  if (norm1 == 0.0f || norm2 == 0.0f) {
    return 0.0f;
  }
  return dot / (norm1 * norm2);
}

static float cosine_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  return 1.0f - cosine_similarity(vec1, vec2);
}

TEST(JitCompiler, vector_cmp) {
  std::vector<int> vec{1, 2, 3};
  Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      x>5&&x<5;
      return 5+x;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<int>, Context&, Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] + 5);
  }
}

TEST(JitCompiler, vector_ternary) {
  std::vector<int> vec{1, 2, 3, 4, 1, 5, 6};
  Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      return x>2?1:0;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<int>, Context&, Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_EQ(result[i], vec[i] > 2 ? 1 : 0);
  }
}

TEST(JitCompiler, vector_string_cmp) {
  std::vector<std::string> left{"hello0", "hello1", "hello2"};
  std::vector<std::string> right{"afasf", "rwrewe", "qw1231241"};

  Context ctx;
  auto simd_left = ctx.NewVector(left);
  auto simd_right = ctx.NewVector(right);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<bit> test_func(Context ctx,simd_vector<string_view> x,simd_vector<string_view> y){
      return x > y;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<Bit>, Context&, Vector<StringView>, Vector<StringView>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, simd_left, simd_right);
  ASSERT_EQ(result.Size(), left.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_EQ(result[i], left[i] > right[i]);
  }
}

TEST(JitCompiler, vector_filter) {
  std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
  std::vector<std::string> citys{"sz", "sh", "bj", "gz", "sh", "bj", "sz", "sh", "gz"};
  Context ctx;
  JitCompiler compiler;

  std::string content = R"(
    filter(ids, citys=="sz")
  )";
  auto rc = compiler.CompileExpression<Vector<int>, Context&, Vector<int>, Vector<StringView>>(content,
                                                                                               {"_", "ids", "citys"});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, {ids, false}, ctx.NewVector(citys));
  ASSERT_EQ(result.Size(), 2);
  ASSERT_EQ(result[0], 10);
  ASSERT_EQ(result[1], 87);
}

static void print_column(Vector<int> c) { RUDF_INFO("print_column:{}", c.Size()); }

RUDF_FUNC_REGISTER(print_column)

TEST(JitCompiler, vector_gather) {
  std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
  Context ctx;
  JitCompiler compiler;

  std::string content = R"(
    void test_func(i32 ctx, simd_vector<i32> ids,simd_vector<i32> scores, simd_vector<i32>
    idxs,simd_vector<i32> idxs1){

      print_column(ids);
      print_column(scores);
       print_column(idxs);
      print_column(idxs1);
    }
  )";

  auto rc = compiler.CompileFunction<void, int32_t, Vector<int>, Vector<int>, Vector<int>, Vector<int>>(content);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(11, ids, ids, ids, ids);
}

TEST(JitCompiler, find) {
  std::string source = R"(
     x.find(100)
  )";

  rapidudf::JitCompiler compiler;
  using simd_vector_f32 = rapidudf::Vector<float>;
  auto result = compiler.CompileExpression<int, simd_vector_f32>(source, {"x"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());

  size_t N = 4096;
  std::vector<float> duration;
  for (size_t i = 0; i < N; i++) {
    duration.emplace_back(i);
  }

  int n = f(duration);
  ASSERT_EQ(n, 100);
}

TEST(JitCompiler, find_gt) {
  std::string source = R"(
     x.find_gt(100)
  )";

  rapidudf::JitCompiler compiler;
  using simd_vector_f32 = rapidudf::Vector<float>;
  auto result = compiler.CompileExpression<int, simd_vector_f32>(source, {"x"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());

  size_t N = 4096;
  std::vector<float> duration;
  for (size_t i = 0; i < N; i++) {
    duration.emplace_back(i);
  }

  int n = f(duration);
  ASSERT_EQ(n, 101);
}

TEST(JitCompiler, example) {
  std::string source = R"(
     x*y + sin(z)
  )";

  rapidudf::JitCompiler compiler({.print_asm = true});
  using simd_vector_f32 = rapidudf::Vector<float>;
  auto result =
      compiler.CompileExpression<simd_vector_f32, Context&, simd_vector_f32, simd_vector_f32, simd_vector_f32>(
          source, {"_", "x", "y", "z"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
}

struct TestParams {
  int boost = 2;
};

static int test_vector_func_unit(int p, TestParams* params) { return p + 100 + params->boost; }

static void test_vector_func(const int* p, TestParams** params, int* output) {
  for (int i = 0; i < kVectorUnitSize; i++) {
    output[i] = test_vector_func_unit(p[i], params[i]);
  }
}

RUDF_VECTOR_FUNC_REGISTER(test_vector_func)

TEST(JitCompiler, user_vector_func) {
  auto* desc = FunctionFactory::GetFunction("test_vector_func");
  ASSERT_TRUE(desc != nullptr);
  ASSERT_TRUE(desc->is_vector_func);

  std::string source = R"(
     x*test_vector_func(y,params)
  )";

  rapidudf::JitCompiler compiler({.print_asm = true});
  using simd_vector_i32 = rapidudf::Vector<int>;
  auto result = compiler.CompileExpression<simd_vector_i32, Context&, simd_vector_i32, simd_vector_i32, TestParams&>(
      source, {"_", "x", "y", "params"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());
  Context ctx;

  std::vector<int> test_x = {1, 2, 3, 4, 5};
  std::vector<int> test_y = {10, 20, 30, 40, 50};
  std::vector<int> test_z(5);
  TestParams params;
  for (size_t i = 0; i < test_x.size(); i++) {
    test_z[i] = test_x[i] * test_vector_func_unit(test_y[i], &params);
  }
  auto z = f(ctx, test_x, test_y, params);
  ASSERT_EQ(z.Size(), test_x.size());

  for (size_t i = 0; i < test_x.size(); i++) {
    ASSERT_EQ(z[i], test_z[i]);
  }
}

TEST(JitCompiler, vector_l2_distance) {
  std::vector<float> vec1, vec2;
  for (size_t i = 0; i < 100; i++) {
    vec1.emplace_back(i + 1.1);
    vec2.emplace_back(i + 1.88);
  }
  float score = l2_distance(vec1, vec2);
  float score1 = functions::simd_vector_l2_distance<float>(vec1, vec2);
  ASSERT_FLOAT_EQ(score, score1);
}

TEST(JitCompiler, vector_cos_distance) {
  std::vector<float> vec1, vec2;
  for (size_t i = 0; i < 124; i++) {
    vec1.emplace_back(i + 1.1);
    vec2.emplace_back(i + 1.58);
  }
  float score = cosine_distance(vec1, vec2);
  float score1 = functions::simd_vector_cosine_distance<float>(vec1, vec2);
  ASSERT_FLOAT_EQ(score, score1);
}
