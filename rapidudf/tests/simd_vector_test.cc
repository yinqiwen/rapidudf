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

#include "rapidudf/types/simd_vector.h"
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <vector>
#include "rapidudf/builtin/simd_vector/ops.h"
#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, vector_cos) {
  std::vector<float> vec{1, 2, 3};
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
      var ret = cos(x);
      return ret;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx, simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  ASSERT_FALSE(result.IsTemporary());
  for (size_t i = 0; i < vec.size(); i++) {
    ASSERT_FLOAT_EQ(result[i], std::cos(vec[i]));
  }
}
TEST(JitCompiler, vector_add) {
  std::vector<float> vec{1, 2, 3};
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
      return x+5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
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
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return x-5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
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
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return 5*x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
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
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return 5/x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
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
  simd::Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      return x%5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
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
  simd::Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      x>5&&x<5;
      return 5+x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
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
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx,simd_vector<f32> x){
      return x+5+10;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
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
  simd::Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(Context ctx,simd_vector<i32> x){
      return x>2?1:0;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>>(content);
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
  auto simd_left = ctx.NewSimdVector(left);
  auto simd_right = ctx.NewSimdVector(right);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<bit> test_func(Context ctx,simd_vector<string_view> x,simd_vector<string_view> y){
      return x > y;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<Bit>, Context&, simd::Vector<StringView>, simd::Vector<StringView>>(
      content);
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
  auto rc = compiler.CompileExpression<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<StringView>>(
      content, {"_", "ids", "citys"});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, ids, ctx.NewSimdVector(citys));
  ASSERT_EQ(result.Size(), 2);
  ASSERT_EQ(result[0], 10);
  ASSERT_EQ(result[1], 87);
}

static void print_column(simd::Vector<int> c) { RUDF_INFO("print_column:{}", c.Size()); }

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

  auto rc = compiler.CompileFunction<void, int32_t, simd::Vector<int>, simd::Vector<int>, simd::Vector<int>,
                                     simd::Vector<int>>(content, true);
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
//       compiler.CompileFunction<simd::Vector<int>, Context&, simd::Vector<int>, simd::Vector<float>,
//       simd::Vector<int>>(
//           content, true);
//   if (!rc.ok()) {
//     RUDF_ERROR("{}", rc.status().ToString());
//   }
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto x = ctx.NewSimdVector(scores);
//   auto idxs = simd::simd_vector_iota<int>(ctx, 0, scores.size());
//   simd::simd_vector_sort_key_value<float, int>(ctx, scores, idxs, true);
//   auto result = simd::simd_vector_gather<int>(ctx, ids, idxs);
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
  simd::Vector<double> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x){
      return pow(x,10);
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<f64>, Context&, simd::Vector<f64>>(content);
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
  auto rc =
      compiler.CompileFunction<simd::Vector<double>, Context&, simd::Vector<double>, simd::Vector<double>, double>(
          source);
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
  rapidudf::simd::Vector<double> Click;
  rapidudf::simd::Vector<double> Like;
  rapidudf::simd::Vector<double> Inter;
  rapidudf::simd::Vector<double> Join;
  rapidudf::simd::Vector<double> TimeV1;
  rapidudf::simd::Vector<double> PostComment;
  rapidudf::simd::Vector<double> PositiveCommentV1;
  rapidudf::simd::Vector<double> ExpoTimeV1;
};
RUDF_STRUCT_FIELDS(Feeds, Click, Like, Inter, Join, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1)
TEST(JitCompiler, test) {
  JitCompiler compiler;
  std::string source = R"(simd_vector<f64> boost_scores(Context ctx,Feeds feeds) {
                              var score = pow(feeds.Click, 10.0);
                              score *= pow(feeds.Like + 0.000082, 4.7);
                              // score *= pow(feeds.Inter, 3.5);
                              // score *= pow(feeds.Join + 0.000024, 5.5);
                              // score *= pow(feeds.TimeV1, 7.0);
                              // score *= pow(feeds.PostComment + 0.000024, 3.5);
                              // score *= pow(feeds.PositiveCommentV1 + 0.0038, 1.0);
                              // score *= pow(feeds.ExpoTimeV1, 1.5);
                              return score;
                            })";
  auto rc = compiler.CompileFunction<rapidudf::simd::Vector<double>, Context&, Feeds&>(source);
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