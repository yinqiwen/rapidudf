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
#include <random>
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
  RUDF_INFO("###dot：{}, left:{}, right:{}", dot, norm1, norm2);
  return dot / (norm1 * norm2);
}

static float cosine_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  float similarity = cosine_similarity(vec1, vec2);
  return 1.0f - similarity;
}

TEST(JitCompiler, vector_add) {
  std::vector<float> vec{1, 2, 3};
  Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
      return x+5;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<float>, Context&, Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  Context ctx;
  auto f = std::move(rc.value());
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] + 5);
  }
}
TEST(JitCompiler, vector_sub) {
  std::vector<float> vec{1, 2, 3};
  Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return x-5;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<float>, Context&, Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] - 5);
  }
}
TEST(JitCompiler, vector_mul) {
  std::vector<float> vec{1, 2, 3};
  Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return 5*x;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<float>, Context&, Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] * 5);
  }
}
TEST(JitCompiler, vector_div) {
  std::vector<float> vec{1, 2, 3};
  Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return 5/x;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<float>, Context&, Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], 5 / vec[i]);
  }
}
TEST(JitCompiler, vector_mod) {
  std::vector<int> vec{1, 2, 3};
  Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      return x%5;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<int>, Context&, Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] % 5);
  }
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

TEST(JitCompiler, vector_add2) {
  std::vector<float> vec{1, 2, 3};
  Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return x+5+10;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<float>, Context&, Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
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
    RUDF_DEBUG("{}", result[i]);
    // ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
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
  auto result = f(ctx, ids, ctx.NewVector(citys));
  ASSERT_EQ(result.Size(), 2);
  ASSERT_EQ(result[0], 10);
  ASSERT_EQ(result[1], 87);
}

static void print_column(Vector<int> c) { RUDF_INFO("print_column:{}", c.Size()); }

RUDF_FUNC_REGISTER(print_column)

TEST(JitCompiler, vector_gather) {
  std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
  std::vector<float> scores{15.6, 22.4, 12.4, 5.6, 333.4, 100.5, 67.8, 1.5, 45.7};
  Context ctx;
  JitCompiler compiler;

  std::string content = R"(
    void test_func(i32 ctx, simd_vector<i32> ids,simd_vector<i32> scores, simd_vector<i32>
    idxs,simd_vector<i32> idxs1){

      print_column(ids);
      print_column(scores);
       print_column(idxs);
      print_column(idxs1);
      // return ids;
    }
  )";

  auto rc = compiler.CompileFunction<void, int32_t, Vector<int>, Vector<int>, Vector<int>, Vector<int>>(content);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  // RUDF_INFO("x size:{}", x.Size());
  f(11, ids, ids, ids, ids);
}

// TEST(JitCompiler, vector_gather) {
//   std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
//   std::vector<float> scores{15.6, 22.4, 12.4, 5.6, 333.4, 100.5, 67.8, 1.5, 45.7};
//   Context ctx;
//   JitCompiler compiler;

//   std::string content = R"(
//     simd_vector<i32> test_func(Context ctx, simd_vector<i32> ids, simd_vector<f32> scores, simd_vector<i32> idxs){
//       // print_column(idxs);
//       // print_column(ids);
//       sort_kv(scores, idxs,true);
//       return gather(ids,idxs);
//     }
//   )";

//   auto rc =
//       compiler.CompileFunction<Vector<int>, Context&, Vector<int>, Vector<float>,
//       Vector<int>>(
//           content);
//   if (!rc.ok()) {
//     RUDF_ERROR("{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto x = ctx.NewVector(scores);
//   auto idxs = simd_vector_iota<int>(ctx, 0, scores.size());
//   simd_vector_sort_key_value<float, int>(ctx, scores, idxs, true);
//   auto result = simd_vector_gather<int>(ctx, ids, idxs);
//   for (size_t i = 0; i < result.Size(); i++) {
//     RUDF_INFO("{} {}", ids[i], scores[i]);
//   }

//   // RUDF_INFO("x size:{}", x.Size());
//   auto result1 = f(ctx, ids, scores, idxs);
//   RUDF_INFO("{}/{}", idxs.Size(), result1.Size());
//   for (size_t i = 0; i < result1.Size(); i++) {
//     RUDF_INFO("{}/{}", result1[i], scores[i]);
//   }
// }

TEST(JitCompiler, vector_pow) {
  std::vector<double> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  Vector<double> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x){
      return pow(x,10);
    }
  )";
  auto rc = compiler.CompileFunction<Vector<f64>, Context&, Vector<f64>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_DOUBLE_EQ(result[i], std::pow(vec[i], 10));
  }
}

TEST(JitCompiler, complex) {
  JitCompiler compiler;
  std::string source = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x,simd_vector<f64> y, double pi){
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<double>, Context&, Vector<double>, Vector<double>, double>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());

  double pi = 3.14159265358979323846264338327950288419716939937510;
  std::vector<double> xx, yy;
  size_t n = 1024;
  for (size_t i = 0; i < n; i++) {
    xx.emplace_back(i + 1);
    yy.emplace_back(i + 101);
  }
  Context ctx;
  auto result = f(ctx, xx, yy, pi);
  ASSERT_EQ(result.Size(), xx.size());
  for (size_t i = 0; i < result.Size(); i++) {
    double x = xx[i];
    double y = yy[i];
    double actual = (x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y);
    ASSERT_DOUBLE_EQ(result[i], actual);
  }
}

struct Feeds {
  rapidudf::Vector<double> Click;
  rapidudf::Vector<double> Like;
  rapidudf::Vector<double> Inter;
  rapidudf::Vector<double> Join;
  rapidudf::Vector<double> TimeV1;
  rapidudf::Vector<double> PostComment;
  rapidudf::Vector<double> PositiveCommentV1;
  rapidudf::Vector<double> ExpoTimeV1;
};
RUDF_STRUCT_FIELDS(Feeds, Click, Like, Inter, Join, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1)
TEST(JitCompiler, test) {
  JitCompiler compiler;
  std::string source = R"(simd_vector<f64> boost_scores(Context ctx,Feeds feeds) {
                              auto score = pow(feeds.Click, 10.0);
                              score *= pow(feeds.Like + 0.000082, 4.7);
                              // score *= pow(feeds.Inter, 3.5);
                              // score *= pow(feeds.Join + 0.000024, 5.5);
                              // score *= pow(feeds.TimeV1, 7.0);
                              // score *= pow(feeds.PostComment + 0.000024, 3.5);
                              // score *= pow(feeds.PositiveCommentV1 + 0.0038, 1.0);
                              // score *= pow(feeds.ExpoTimeV1, 1.5);
                              return score;
                            })";
  auto rc = compiler.CompileFunction<rapidudf::Vector<double>, Context&, Feeds&>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto& stat = f.Stats();
  RUDF_INFO("parse cost:{}us, parse_validate cost:{}us, IR_build cost:{}us, jit compile cost:{}us",
            stat.parse_cost.count(), stat.parse_validate_cost.count(), stat.ir_build_cost.count(),
            stat.compile_cost.count());

  std::vector<double> clicks;
  std::vector<double> likes;
  for (size_t i = 0; i < 100; i++) {
    clicks.emplace_back(i + 11);
    likes.emplace_back(i + 12);
  }
  Feeds feeds;
  feeds.Click = clicks;
  feeds.Like = likes;
  Context ctx;
  auto result = f(ctx, feeds);
  for (size_t i = 0; i < result.Size(); i++) {
    double actual_score = std::pow(clicks[i], 10.0);
    actual_score *= std::pow(likes[i] + 0.000082, 4.7);
    ASSERT_DOUBLE_EQ(result[i], actual_score);
  }
}

static float __attribute__((noinline)) get_duration_score(float duration, float alpha, float beta) {
  float x = (duration - alpha) / beta;
  return 1.0 / (1 + std::exp(-x));
}

TEST(JitCompiler, duration_score) {
  std::string source = R"(
    simd_vector<f32> get_duration_score(Context ctx, simd_vector<f32> duration, f32 alpha, f32 beta)
    {
        auto x = (duration-alpha)/beta;
        return 1.0_f32/(1_f32 + exp(-x));
    }
  )";

  rapidudf::JitCompiler compiler;
  rapidudf::Context ctx;
  using simd_vector_f32 = rapidudf::Vector<float>;

  auto result = compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, float, float>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());

  size_t N = 4096;
  std::vector<float> duration;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(1, 100);
  for (size_t i = 0; i < N; i++) {
    int v = static_cast<int>(distr(gen));
    duration.emplace_back(static_cast<float>(v));
  }

  float alpha = 30000.0;
  float beta = 10000.0;
  f(ctx, duration, alpha, beta);

  auto start_time = std::chrono::high_resolution_clock::now();
  auto vector_result = f(ctx, duration, alpha, beta);
  auto vector_compute_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

  std::vector<float> normal_results(duration.size());
  start_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; i++) {
    normal_results[i] = get_duration_score(duration[i], alpha, beta);
  }
  auto normal_compute_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  RUDF_INFO("Normal compute cost {}us, vector compute cost {}us", normal_compute_duration.count(),
            vector_compute_duration.count());
  for (size_t i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(vector_result[i], normal_results[i]);
  }
}

static float __attribute__((noinline)) wilson_ctr(float exp_cnt, float clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * std::sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
}

TEST(JitCompiler, wilson_ctr) {
  std::string source = R"(
    simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> exp_cnt, simd_vector<f32> clk_cnt)
    {
       return log10(exp_cnt) *
         (clk_cnt / exp_cnt +  1.96_f32 * 1.96_f32 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);

    }
  )";

  rapidudf::JitCompiler compiler;
  rapidudf::Context ctx;
  using simd_vector_f32 = rapidudf::Vector<float>;

  auto result = compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, simd_vector_f32>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
  }
  ASSERT_TRUE(result.ok());
  auto f = std::move(result.value());

  size_t N = 40990;
  std::vector<float> exp_cnt;
  std::vector<float> clk_cnt;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(1, 100);
  for (size_t i = 0; i < N; i++) {
    int v = static_cast<int>(distr(gen));
    clk_cnt.emplace_back(static_cast<float>(v));
    v += 10;
    exp_cnt.emplace_back(static_cast<float>(v));
  }

  f(ctx, exp_cnt, clk_cnt);

  auto start_time = std::chrono::high_resolution_clock::now();
  auto vector_wilson_ctr_result = f(ctx, exp_cnt, clk_cnt);
  auto vector_compute_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

  std::vector<float> normal_results(exp_cnt.size());
  start_time = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; i++) {
    normal_results[i] = wilson_ctr(exp_cnt[i], clk_cnt[i]);
  }
  auto normal_compute_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  RUDF_INFO("Normal compute cost {}us, vector compute cost {}us", normal_compute_duration.count(),
            vector_compute_duration.count());
  // for (size_t i = 0; i < N; i++) {
  //   ASSERT_FLOAT_EQ(vector_wilson_ctr_result[i], normal_results[i]);
  // }
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
  RUDF_INFO("{} {}", score, score1);
  ASSERT_FLOAT_EQ(score, score1);
}
