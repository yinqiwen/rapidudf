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
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

// TEST(JitCompiler, vector_cos) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
//       var ret = cos(x);
//       return ret;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   ASSERT_FALSE(result.IsTemporary());
//   for (size_t i = 0; i < vec.size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], std::cos(vec[i]));
//   }
// }
// TEST(JitCompiler, vector_add) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
//       return x+5;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   Context ctx;
//   auto f = std::move(rc.value());
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] + 5);
//   }
// }
// TEST(JitCompiler, vector_sub) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
//       return x-5;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] - 5);
//   }
// }
// TEST(JitCompiler, vector_mul) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
//       return 5*x;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] * 5);
//   }
// }
// TEST(JitCompiler, vector_div) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
//       return 5/x;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], 5 / vec[i]);
//   }
// }
// TEST(JitCompiler, vector_mod) {
//   std::vector<int> vec{1, 2, 3};
//   simd::Vector<int> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
//       return x%5;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] % 5);
//   }
// }

// TEST(JitCompiler, vector_cmp) {
//   std::vector<int> vec{1, 2, 3};
//   simd::Vector<int> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
//       x>5&&x<5;
//       return 5+x;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] + 5);
//   }
// }

// TEST(JitCompiler, vector_add2) {
//   std::vector<float> vec{1, 2, 3};
//   simd::Vector<float> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
//       return x+5+10;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
//   }
// }

// TEST(JitCompiler, vector_ternary) {
//   std::vector<int> vec{1, 2, 3, 4, 1, 5, 6};
//   simd::Vector<int> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
//       return x>2?1:0;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     RUDF_DEBUG("{}", result[i]);
//     // ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
//   }
// }

// TEST(JitCompiler, vector_string_cmp) {
//   std::vector<std::string> left{"hello0", "hello1", "hello2"};
//   std::vector<std::string> right{"afasf", "rwrewe", "qw1231241"};

//   Context ctx;
//   auto simd_left = ctx.NewSimdVector(left);
//   auto simd_right = ctx.NewSimdVector(right);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<bit> test_func(Context ctx,simd_vector<string_view> x,simd_vector<string_view> y){
//       return x > y;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<Bit>, Context&, simd::Vector<StringView>,
//   simd::Vector<StringView>>(
//       content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto result = f(ctx, simd_left, simd_right);
//   ASSERT_EQ(result.Size(), left.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_EQ(result[i], left[i] > right[i]);
//   }
// }

// TEST(JitCompiler, vector_filter) {
//   std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
//   std::vector<std::string> citys{"sz", "sh", "bj", "gz", "sh", "bj", "sz", "sh", "gz"};
//   Context ctx;
//   JitCompiler compiler;

//   std::string content = R"(
//     filter(ids, citys=="sz")
//   )";
//   auto rc = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<StringView>>(
//       content, {"_", "ids", "citys"});
//   if (!rc.ok()) {
//     RUDF_ERROR("{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto result = f(ctx, ids, ctx.NewSimdVector(citys));
//   ASSERT_EQ(result.Size(), 2);
//   ASSERT_EQ(result[0], 10);
//   ASSERT_EQ(result[1], 87);
// }

// static void print_column(simd::Vector<int> c) { RUDF_INFO("print_column:{}", c.Size()); }

// RUDF_FUNC_REGISTER(print_column)

// TEST(JitCompiler, vector_gather) {
//   std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
//   std::vector<float> scores{15.6, 22.4, 12.4, 5.6, 333.4, 100.5, 67.8, 1.5, 45.7};
//   Context ctx;
//   JitCompiler compiler;

//   std::string content = R"(
//     void test_func(i32 ctx, simd_vector<i32> ids,simd_vector<i32> scores, simd_vector<i32>
//     idxs,simd_vector<i32> idxs1){

//       print_column(ids);
//       print_column(scores);
//        print_column(idxs);
//       print_column(idxs1);
//       // return ids;
//     }
//   )";

//   auto rc = compiler.CompileFunction<void, int32_t, simd::Vector<int>, simd::Vector<int>, simd::Vector<int>,
//                                      simd::Vector<int>>(content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   // RUDF_INFO("x size:{}", x.Size());
//   f(11, ids, ids, ids, ids);
// }

// // TEST(JitCompiler, vector_gather) {
// //   std::vector<int> ids{10, 11, 23, 45, 67, 88, 87, 99, 15};
// //   std::vector<float> scores{15.6, 22.4, 12.4, 5.6, 333.4, 100.5, 67.8, 1.5, 45.7};
// //   Context ctx;
// //   JitCompiler compiler;

// //   std::string content = R"(
// //     simd_vector<i32> test_func(Context ctx, simd_vector<i32> ids, simd_vector<f32> scores, simd_vector<i32> idxs){
// //       // print_column(idxs);
// //       // print_column(ids);
// //       sort_kv(scores, idxs,true);
// //       return gather(ids,idxs);
// //     }
// //   )";

// //   auto rc =
// //       compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<float>,
// //       simd::Vector<int>>(
// //           content, true);
// //   if (!rc.ok()) {
// //     RUDF_ERROR("{}", rc.status().ToString());
// //   }
// //   ASSERT_TRUE(rc.ok());
// //   auto f = std::move(rc.value());
// //   auto x = ctx.NewSimdVector(scores);
// //   auto idxs = simd::simd_vector_iota<int>(ctx, 0, scores.size());
// //   simd::simd_vector_sort_key_value<float, int>(ctx, scores, idxs, true);
// //   auto result = simd::simd_vector_gather<int>(ctx, ids, idxs);
// //   for (size_t i = 0; i < result.Size(); i++) {
// //     RUDF_INFO("{} {}", ids[i], scores[i]);
// //   }

// //   // RUDF_INFO("x size:{}", x.Size());
// //   auto result1 = f(ctx, ids, scores, idxs);
// //   RUDF_INFO("{}/{}", idxs.Size(), result1.Size());
// //   for (size_t i = 0; i < result1.Size(); i++) {
// //     RUDF_INFO("{}/{}", result1[i], scores[i]);
// //   }
// // }

// TEST(JitCompiler, vector_pow) {
//   std::vector<double> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//   simd::Vector<double> simd_vec(vec);
//   JitCompiler compiler;
//   std::string content = R"(
//     simd_vector<f64> test_func(Context ctx, simd_vector<f64> x){
//       return pow(x,10);
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<f64>, Context&, simd::Vector<f64>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   Context ctx;
//   auto result = f(ctx, simd_vec);
//   ASSERT_EQ(result.Size(), vec.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_DOUBLE_EQ(result[i], std::pow(vec[i], 10));
//   }
// }

// TEST(JitCompiler, complex) {
//   JitCompiler compiler;
//   std::string source = R"(
//     simd_vector<f64> test_func(Context ctx, simd_vector<f64> x,simd_vector<f64> y, double pi){
//       return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
//     }
//   )";
//   auto rc =
//       compiler.CompileFunction<simd::Vector<double>, Context&, simd::Vector<double>, simd::Vector<double>, double>(
//           source);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());

//   double pi = 3.14159265358979323846264338327950288419716939937510;
//   std::vector<double> xx, yy;
//   size_t n = 1024;
//   for (size_t i = 0; i < n; i++) {
//     xx.emplace_back(i + 1);
//     yy.emplace_back(i + 101);
//   }
//   Context ctx;
//   auto result = f(ctx, xx, yy, pi);
//   ASSERT_EQ(result.Size(), xx.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     double x = xx[i];
//     double y = yy[i];
//     double actual = (x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y);
//     ASSERT_DOUBLE_EQ(result[i], actual);
//   }
// }

// struct Feeds {
//   rapidudf::simd::Vector<double> Click;
//   rapidudf::simd::Vector<double> Like;
//   rapidudf::simd::Vector<double> Inter;
//   rapidudf::simd::Vector<double> Join;
//   rapidudf::simd::Vector<double> TimeV1;
//   rapidudf::simd::Vector<double> PostComment;
//   rapidudf::simd::Vector<double> PositiveCommentV1;
//   rapidudf::simd::Vector<double> ExpoTimeV1;
// };
// RUDF_STRUCT_FIELDS(Feeds, Click, Like, Inter, Join, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1)
// TEST(JitCompiler, test) {
//   JitCompiler compiler;
//   std::string source = R"(simd_vector<f64> boost_scores(Context ctx,Feeds feeds) {
//                               var score = pow(feeds.Click, 10.0);
//                               score *= pow(feeds.Like + 0.000082, 4.7);
//                               // score *= pow(feeds.Inter, 3.5);
//                               // score *= pow(feeds.Join + 0.000024, 5.5);
//                               // score *= pow(feeds.TimeV1, 7.0);
//                               // score *= pow(feeds.PostComment + 0.000024, 3.5);
//                               // score *= pow(feeds.PositiveCommentV1 + 0.0038, 1.0);
//                               // score *= pow(feeds.ExpoTimeV1, 1.5);
//                               return score;
//                             })";
//   auto rc = compiler.CompileFunction<rapidudf::simd::Vector<double>, Context&, Feeds&>(source);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto& stat = f.Stats();
//   RUDF_INFO("parse cost:{}us, parse_validate cost:{}us, IR_build cost:{}us, jit compile cost:{}us",
//             stat.parse_cost.count(), stat.parse_validate_cost.count(), stat.ir_build_cost.count(),
//             stat.compile_cost.count());

//   std::vector<double> clicks;
//   std::vector<double> likes;
//   for (size_t i = 0; i < 100; i++) {
//     clicks.emplace_back(i + 11);
//     likes.emplace_back(i + 12);
//   }
//   Feeds feeds;
//   feeds.Click = clicks;
//   feeds.Like = likes;
//   Context ctx;
//   auto result = f(ctx, feeds);
//   for (size_t i = 0; i < result.Size(); i++) {
//     double actual_score = std::pow(clicks[i], 10.0);
//     actual_score *= std::pow(likes[i] + 0.000082, 4.7);
//     ASSERT_DOUBLE_EQ(result[i], actual_score);
//   }
// }

static float __attribute__((noinline)) get_duration_score(float duration, float alpha, float beta) {
  float x = (duration - alpha) / beta;
  return 1.0 / (1 + std::exp(-x));
}

// TEST(JitCompiler, duration_score) {
//   std::string source = R"(
//     simd_vector<f32> get_duration_score(Context ctx, simd_vector<f32> duration, f32 alpha, f32 beta)
//     {
//         var x = (duration-alpha)/beta;
//         return 1.0_f32/(1_f32 + exp(-x));
//     }
//   )";

//   rapidudf::JitCompiler compiler;
//   rapidudf::Context ctx;
//   using simd_vector_f32 = rapidudf::simd::Vector<float>;

//   auto result =
//       compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, float, float>(source, true);
//   if (!result.ok()) {
//     RUDF_ERROR("{}", result.status().ToString());
//   }
//   ASSERT_TRUE(result.ok());
//   auto f = std::move(result.value());

//   size_t N = 4096;
//   std::vector<float> duration;
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> distr(1, 100);
//   for (size_t i = 0; i < N; i++) {
//     int v = static_cast<int>(distr(gen));
//     duration.emplace_back(static_cast<float>(v));
//   }

//   float alpha = 30000.0;
//   float beta = 10000.0;
//   f(ctx, duration, alpha, beta);

//   auto start_time = std::chrono::high_resolution_clock::now();
//   auto vector_result = f(ctx, duration, alpha, beta);
//   auto vector_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   std::vector<float> normal_results(duration.size());
//   start_time = std::chrono::high_resolution_clock::now();
//   for (size_t i = 0; i < N; i++) {
//     normal_results[i] = get_duration_score(duration[i], alpha, beta);
//   }
//   auto normal_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
//   RUDF_INFO("Normal compute cost {}us, vector compute cost {}us", normal_compute_duration.count(),
//             vector_compute_duration.count());
//   for (size_t i = 0; i < N; i++) {
//     ASSERT_FLOAT_EQ(vector_result[i], normal_results[i]);
//   }
// }

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
      // return  1.96 / (2 * exp_cnt);

    }
  )";

  rapidudf::JitCompiler compiler;
  rapidudf::Context ctx;
  using simd_vector_f32 = rapidudf::simd::Vector<float>;

  auto result =
      compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, simd_vector_f32>(source, true);
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
  for (size_t i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(vector_wilson_ctr_result[i], normal_results[i]);
  }
}

// TEST(JitCompiler, fused) {
//   size_t N = 40966;
//   std::vector<float> a;
//   std::vector<float> b;
//   std::vector<float> c;
//   std::vector<float> d;
//   std::vector<float> e;
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> distr(1, 100);
//   for (size_t i = 0; i < N; i++) {
//     int v = static_cast<int>(distr(gen));
//     a.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     b.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     c.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     d.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     e.emplace_back(static_cast<float>(v));
//   }
//   rapidudf::Context ctx;

//   std::vector<EvalValue> rpn;
//   // // a b * b c / +
//   rpn.emplace_back(to_eval_value(simd::Vector<float>(a)));
//   rpn.emplace_back(to_eval_value(simd::Vector<float>(a)));
//   rpn.emplace_back(OP_MULTIPLY);
//   rpn.emplace_back(to_eval_value(simd::Vector<float>(c)));
//   rpn.emplace_back(OP_PLUS);
//   rpn.emplace_back(to_eval_value(simd::Vector<float>(a)));
//   rpn.emplace_back(OP_PLUS);
//   rpn.emplace_back(to_eval_value(simd::Vector<float>(c)));
//   rpn.emplace_back(OP_MINUS);
//   // rpn.emplace_back(OP_SIN);
//   absl::Span<EvalValue> rpn_view(rpn);
//   size_t test_count = 10;

//   std::vector<float> normal_results(a.size());
//   auto start_time = std::chrono::high_resolution_clock::now();
//   for (size_t k = 0; k < test_count; k++) {
//     for (size_t i = 0; i < N; i++) {
//       normal_results[i] = (a[i] * a[i] + c[i] + a[i] - c[i]);
//     }
//   }

//   auto normal_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   std::string expr = "(a*a+c+a-c)";
//   JitCompiler compiler;
//   auto result =
//       compiler.CompileExpression<simd::Vector<float>, rapidudf::Context&, simd::Vector<float>, simd::Vector<float>,
//                                  simd::Vector<float>, simd::Vector<float>, simd::Vector<float>>(
//           expr, {"_", "a", "b", "c", "d", "e"}, true);
//   if (!result.ok()) {
//     RUDF_ERROR("{}", result.status().ToString());
//   }
//   ASSERT_TRUE(result.ok());
//   auto f = std::move(result.value());
//   auto vector_result = f(ctx, a, b, c, d, e);
//   start_time = std::chrono::high_resolution_clock::now();
//   for (size_t k = 0; k < test_count; k++) {
//     ctx.Reset();
//     vector_result = f(ctx, a, b, c, d, e);
//   }
//   auto vector2_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   for (size_t i = 0; i < N; i++) {
//     ASSERT_FLOAT_EQ(vector_result[i], normal_results[i]);
//   }

//   ctx.Reset();
//   auto fused_vector_result = simd::simd_vector_eval<float>(ctx, rpn_view);
//   start_time = std::chrono::high_resolution_clock::now();
//   for (size_t k = 0; k < test_count; k++) {
//     ctx.Reset();
//     fused_vector_result = simd::simd_vector_eval<float>(ctx, rpn_view);
//   }
//   auto vector_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   RUDF_INFO("Normal compute cost {}us, fused vector compute cost {}us, vector compute cost {}us",
//             normal_compute_duration.count(), vector_compute_duration.count(), vector2_compute_duration.count());
//   auto fresult = fused_vector_result;
//   for (size_t i = 0; i < N; i++) {
//     ASSERT_FLOAT_EQ(fresult[i], normal_results[i]);
//   }

//   EvalValue x;
//   simd::VectorData xv(nullptr, 123);
//   x.vector = xv;

//   RUDF_INFO("null:{}", x.op);
// }

// TEST(JitCompiler, fused1) {
//   size_t N = 4099;
//   std::vector<float> a;
//   std::vector<float> b;
//   std::vector<float> c;

//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> distr(1, 100);
//   for (size_t i = 0; i < N; i++) {
//     int v = static_cast<int>(distr(gen));
//     a.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     b.emplace_back(static_cast<float>(v));
//     v = static_cast<int>(distr(gen));
//     c.emplace_back(static_cast<float>(v));
//   }
//   rapidudf::Context ctx;

//   size_t test_count = 100;

//   std::vector<float> normal_results(a.size());
//   auto start_time = std::chrono::high_resolution_clock::now();
//   for (size_t k = 0; k < test_count; k++) {
//     for (size_t i = 0; i < N; i++) {
//       normal_results[i] = a[i] * b[i] + c[i];
//     }
//   }

//   auto normal_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   std::string expr = "a*b+c";
//   JitCompiler compiler;
//   auto result = compiler.CompileExpression<simd::Vector<float>, rapidudf::Context&, simd::Vector<float>,
//                                            simd::Vector<float>, simd::Vector<float>>(expr, {"_", "a", "b", "c"},
//                                            true);
//   if (!result.ok()) {
//     RUDF_ERROR("{}", result.status().ToString());
//   }
//   ASSERT_TRUE(result.ok());
//   auto f = std::move(result.value());
//   auto vector_result = f(ctx, a, b, c);
//   start_time = std::chrono::high_resolution_clock::now();
//   for (size_t k = 0; k < test_count; k++) {
//     ctx.Reset();
//     vector_result = f(ctx, a, b, c);
//   }
//   auto vector_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   ctx.Reset();

//   std::string expr1 = "fma(a,b,c)";
//   auto result1 =
//       compiler.CompileExpression<simd::Vector<float>, rapidudf::Context&, simd::Vector<float>, simd::Vector<float>,
//                                  simd::Vector<float>>(expr1, {"_", "a", "b", "c"}, true);
//   ASSERT_TRUE(result1.ok());
//   auto f1 = std::move(result1.value());

//   auto fused_vector_result = f1(ctx, a, b, c);
//   start_time = std::chrono::high_resolution_clock::now();
//   // auto fused_vector_result = simd::simd_vector_ternary_op<float, float, rapidudf::OP_FMA>(ctx, a, b, c);
//   for (size_t k = 0; k < test_count; k++) {
//     ctx.Reset();
//     fused_vector_result = f1(ctx, a, b, c);
//   }
//   auto fused_vector_compute_duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);

//   RUDF_INFO("fused vector compute cost {}us, vector compute cost {}us", fused_vector_compute_duration.count(),
//             vector_compute_duration.count());
// }