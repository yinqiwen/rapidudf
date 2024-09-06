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
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/simd.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;

TEST(JitCompiler, vector_cos) {
  std::vector<float> vec{1, 2, 3};
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(simd_vector<f32> x){
      return cos(x);
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  // ASSERT_EQ(f(simd_vec), vec.size());
}
TEST(JitCompiler, vector_add) {
  std::vector<float> vec{1, 2, 3};
  simd::Vector<float> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<f32> test_func(simd_vector<f32> x){
      return x+5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<f32> test_func(simd_vector<f32> x){
      return x-5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<f32> test_func(simd_vector<f32> x){
      return 5*x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<f32> test_func(simd_vector<f32> x){
      return 5/x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<i32> test_func(simd_vector<i32> x){
      return x%5;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, simd::Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<i32> test_func(simd_vector<i32> x){
      x>5&&x<5;
      return 5+x;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, simd::Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
    simd_vector<f32> test_func(simd_vector<f32> x){
      return x+5+10;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<float>, simd::Vector<float>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
  }
}

TEST(JitCompiler, vector_ternary) {
  std::vector<int> vec{1, 2, 3, 4, 1, 5, 6};
  simd::Vector<int> simd_vec(vec);
  JitCompiler compiler(4096, true);
  std::string content = R"(
    simd_vector<i32> test_func(simd_vector<i32> x){
      return x>2?1:0;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, simd::Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
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
  JitCompiler compiler(4096, true);
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
  JitCompiler compiler(4096, true);
  std::string content = R"(
    simd_vector<f64> test_func(){
      var t = iota(1_f64,12);
      return t;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<double>>(content, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f();
  RUDF_INFO("IsTemporary:{}", result.IsTemporary());
  ASSERT_EQ(result.Size(), 12);
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_DOUBLE_EQ(result[i], i + 1);
  }

  DType d(DATA_VOID);
  DType d1(DATA_U8);
  RUDF_INFO("can cast to:{}", d.CanCastTo(d1));
}

// TEST(JitCompiler, vector_string_cmp) {
//   std::vector<std::string> left{"hello0", "hello1", "hello2"};
//   std::vector<std::string> right{"afasf", "rwrewe", "qw1231241"};
//   auto left_views = StringView::makeVector(left);
//   auto right_views = StringView::makeVector(right);
//   simd::Vector<StringView> simd_left(left_views);
//   simd::Vector<StringView> simd_right(right_views);
//   JitCompiler compiler(4096, true);
//   std::string content = R"(
//     simd_vector<bit> test_func(simd_vector<string_view> x,simd_vector<string_view> y){
//       return x > y;
//     }
//   )";
//   auto rc = compiler.CompileFunction<simd::Vector<Bit>, simd::Vector<StringView>, simd::Vector<StringView>>(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   auto result = f(simd_left, simd_right);
//   ASSERT_EQ(result.Size(), left.size());
//   for (size_t i = 0; i < result.Size(); i++) {
//     ASSERT_EQ(result[i], left[i] > right[i]);
//   }
// }
