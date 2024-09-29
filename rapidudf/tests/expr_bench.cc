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
#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>
#include "exprtk.hpp"
#include "rapidudf/context/context.h"
#include "rapidudf/rapidudf.h"

static size_t test_n = 1024;
static const double pi = 3.14159265358979323846264338327950288419716939937510;
static std::vector<double> xx, yy, actuals, final_results;
static double xi, yi;

static void init_test_numbers() {
  xx.clear();
  yy.clear();
  actuals.clear();
  final_results.clear();
  final_results.resize(test_n);
  for (size_t i = 0; i < test_n; i++) {
    xx.emplace_back(i + 1);
    yy.emplace_back(i + 101);
    double x = xx[i];
    double y = yy[i];
    double actual = x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    actuals.emplace_back(actual);
  }
}

static rapidudf::JitFunction<double, double, double, double> g_expr_func;
static rapidudf::JitFunction<rapidudf::simd::Vector<double>, rapidudf::Context&, rapidudf::simd::Vector<double>,
                             rapidudf::simd::Vector<double>>
    g_vector_expr_func;

static void DoRapidUDFExprSetup(const benchmark::State& state) {
  std::string source = R"(
    double test_func(double x,double y, double pi){
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )";
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<double, double, double, double>(source, false);
  g_expr_func = std::move(result.value());
  init_test_numbers();
}

static void DoRapidUDFExprTeardown(const benchmark::State& state) {}

static void BM_rapidudf_expr_func(benchmark::State& state) {
  double results = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < test_n; i++) {
      double x = xx[i];
      double y = yy[i];
      double result = g_expr_func(x, y, pi);
      results += result;
    }
  }
  RUDF_INFO("RapidUDF result:{}", results);
  // double result = 0;
  // size_t n = 0;
  // for (auto _ : state) {
  //   result += g_expr_func(100, 102, pi);
  //   n++;
  // }
  // double x = 100;
  // double y = 102;
  // double actual = x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
  // RUDF_INFO("Result:{} for loop:{}, actual:{}", result, n, actual);
}
BENCHMARK(BM_rapidudf_expr_func)->Setup(DoRapidUDFExprSetup)->Teardown(DoRapidUDFExprTeardown);

static void DoRapidUDFVectorExprSetup(const benchmark::State& state) {
  std::string source = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x,simd_vector<f64> y){
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )";
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<rapidudf::simd::Vector<double>, rapidudf::Context&,
                                         rapidudf::simd::Vector<double>, rapidudf::simd::Vector<double>>(source, true);

  g_vector_expr_func = std::move(result.value());

  init_test_numbers();
}

static void DoRapidUDFVectorExprTeardown(const benchmark::State& state) {}

static void BM_rapidudf_vector_expr_func(benchmark::State& state) {
  rapidudf::Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto results = g_vector_expr_func(ctx, xx, yy);
    // for (size_t i = 0; i < results.Size(); i++) {
    //  if (results[i] != actuals[i]) {
    //    RUDF_INFO("[{}]Error result:{}, while expected:{}", i, results[i], actuals[i]);
    //  }
    //}
  }
}
BENCHMARK(BM_rapidudf_vector_expr_func)->Setup(DoRapidUDFVectorExprSetup)->Teardown(DoRapidUDFVectorExprTeardown);

static exprtk::parser<double> exprtk_parser;
static exprtk::expression<double> exprtk_expression;
static exprtk::symbol_table<double> exprtk_symbol_table;
static void DoExprtkExprSetup(const benchmark::State& state) {
  exprtk_symbol_table.add_constants();
  exprtk_symbol_table.add_variable("x", xi);
  exprtk_symbol_table.add_variable("y", yi);
  // exprtk_symbol_table.add_vector("results", final_results);
  exprtk_expression.register_symbol_table(exprtk_symbol_table);
  std::string expr = "x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y";
  // std::string expr = "cos(y - sin(2 / x * pi))";
  if (!exprtk_parser.compile(expr, exprtk_expression)) {
    RUDF_ERROR("[load_expression] - Parser Error:{}\tExpression: {}", exprtk_parser.error().c_str(), expr);
  }
  init_test_numbers();
}

static void DooExprtkExprTeardown(const benchmark::State& state) {}

static void BM_exprtk_expr_func(benchmark::State& state) {
  double results = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < test_n; i++) {
      xi = xx[i];
      yi = yy[i];
      results += exprtk_expression.value();
    }
  }
  RUDF_INFO("Exprtk result:{}", results);
}
BENCHMARK(BM_exprtk_expr_func)->Setup(DoExprtkExprSetup)->Teardown(DooExprtkExprTeardown);

static void DoNativeFuncSetup(const benchmark::State& state) { init_test_numbers(); }

static void DoNativeFuncTeardown(const benchmark::State& state) {}

static void BM_native_func(benchmark::State& state) {
  double results = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < test_n; i++) {
      double x = xx[i];
      double y = yy[i];
      double result = x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
      results += result;
    }
  }
  RUDF_INFO("Native result:{}", results);
}
BENCHMARK(BM_native_func)->Setup(DoNativeFuncSetup)->Teardown(DoNativeFuncTeardown);

BENCHMARK_MAIN();