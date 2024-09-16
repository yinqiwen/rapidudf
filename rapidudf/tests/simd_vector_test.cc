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
#include "rapidudf/context/context.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, vector_cos) {
  std::vector<float> vec{1, 2, 3};
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(Context ctx, simd_vector<f32> x){
      return cos(x);
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, Context&, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  ASSERT_EQ(f(ctx, simd_vec).Size(), vec.size());
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
TEST(JitCompiler, vector_dot) {
  std::vector<float> left{1, 2, 3, 4, 1, 5, 6};
  std::vector<float> right{10, 20, 30, 40, 10, 50, 60};
  simd::Vector<float> simd_left(left);
  simd::Vector<float> simd_right(right);
  JitCompiler compiler;
  std::string content = R"(
    f32 test_func(simd_vector<f32> x,simd_vector<f32> y){
      return dot(x,y);
    }
  )";
  auto rc = compiler.CompileFunction<float, simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_left, simd_right);
  float native_result = 0;
  for (size_t i = 0; i < left.size(); i++) {
    native_result += (left[i] * right[i]);
  }
  ASSERT_FLOAT_EQ(result, native_result);
}

TEST(JitCompiler, vector_iota) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f64> test_func(Context ctx){
      var t = iota(1_f64,12);
      return t;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<double>, Context&>(content, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  auto result = f(ctx);
  RUDF_INFO("IsTemporary:{}", result.IsTemporary());
  ASSERT_EQ(result.Size(), 12);
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_DOUBLE_EQ(result[i], i + 1);
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