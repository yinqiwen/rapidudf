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
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, vector_size) {
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
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
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  simd::Vector<int> simd_vec(vec);
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<i32> test_func(simd_vector<i32> x){
      return x+5+10;
    }
  )";
  auto rc = compiler.CompileFunction<simd::Vector<int>, simd::Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(simd_vec);
  ASSERT_EQ(result.Size(), vec.size());
  for (size_t i = 0; i < result.Size(); i++) {
    ASSERT_FLOAT_EQ(result[i], vec[i] + 5 + 10);
  }
}

TEST(JitCompiler, vector_ternary) {
  spdlog::set_level(spdlog::level::debug);
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
