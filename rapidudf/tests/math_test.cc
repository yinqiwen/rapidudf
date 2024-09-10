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
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;

TEST(JitCompiler, sqrt) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
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

TEST(JitCompiler, complex_math) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;
  ParseContext ctx;
  std::string content = R"(
    double test_func(double a, double t, double c){
      return a * exp(2.2/3.3*t) + c;
    }
  )";
  auto rc = compiler.CompileFunction<double, double, double, double>(content, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  double a = 1.2;
  double t = 2.1;
  double c = 3.3;

  double v = a * std::exp(2.2 / 3.3 * t) + c;
  ASSERT_DOUBLE_EQ(f(a, t, c), v);
}
