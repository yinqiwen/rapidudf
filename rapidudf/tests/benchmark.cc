/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/rapidudf.h"

static size_t test_n = 4099;
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
  auto result = compiler.CompileFunction<double, double, double, double>(source);
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
  RUDF_DEBUG("RapidUDF result:{}", results);
  //  double result = 0;
  //  size_t n = 0;
  //  for (auto _ : state) {
  //    result += g_expr_func(100, 102, pi);
  //    n++;
  //  }
  //  double x = 100;
  //  double y = 102;
  //  double actual = x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
  //  RUDF_INFO("Result:{} for loop:{}, actual:{}", result, n, actual);
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
                                         rapidudf::simd::Vector<double>, rapidudf::simd::Vector<double>>(source);

  g_vector_expr_func = std::move(result.value());

  init_test_numbers();
}

static void DoRapidUDFVectorExprTeardown(const benchmark::State& state) {}

static void BM_rapidudf_vector_expr_func(benchmark::State& state) {
  rapidudf::Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto results = g_vector_expr_func(ctx, xx, yy);
    RUDF_DEBUG("size:{}", results.Size());
  }
}
BENCHMARK(BM_rapidudf_vector_expr_func)->Setup(DoRapidUDFVectorExprSetup)->Teardown(DoRapidUDFVectorExprTeardown);

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
  RUDF_DEBUG("Native result:{}", results);
}
BENCHMARK(BM_native_func)->Setup(DoNativeFuncSetup)->Teardown(DoNativeFuncTeardown);

static float __attribute__((noinline)) wilson_ctr(float exp_cnt, float clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * std::sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
}

static std::vector<float> exp_cnt;
static std::vector<float> clk_cnt;
static std::random_device rd;
static std::mt19937 gen(rd());

static void native_wilson_ctr_setup(const benchmark::State& state) {
  clk_cnt.clear();
  exp_cnt.clear();
  std::uniform_int_distribution<> distr(1, 100);
  for (size_t i = 0; i < test_n; i++) {
    int v = static_cast<int>(distr(gen));
    clk_cnt.emplace_back(static_cast<float>(v));
    v += 10;
    exp_cnt.emplace_back(static_cast<float>(v));
  }
}

static void native_wilson_ctr_teardown(const benchmark::State& state) {}

static void BM_native_wilson_ctr(benchmark::State& state) {
  double results = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < test_n; i++) {
      results += wilson_ctr(exp_cnt[i], clk_cnt[i]);
    }
  }
  RUDF_DEBUG("Native _wilson_ctr result:{}", results);
}
BENCHMARK(BM_native_wilson_ctr)->Setup(native_wilson_ctr_setup)->Teardown(native_wilson_ctr_teardown);

using simd_vector_f32 = rapidudf::simd::Vector<float>;
static rapidudf::JitFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, simd_vector_f32>
    g_vector_wilson_ctr_func;

static void rapidudf_vector_wilson_ctr_setup(const benchmark::State& state) {
  native_wilson_ctr_setup(state);
  std::string source = R"(
    simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> exp_cnt, simd_vector<f32> clk_cnt)
    {
       return log10(exp_cnt) *
         (clk_cnt / exp_cnt +  1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) * sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
    }
  )";
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, simd_vector_f32>(source);
  g_vector_wilson_ctr_func = std::move(result.value());
}

static void BM_rapidudf_vector_wilson_ctr(benchmark::State& state) {
  double results = 0;
  rapidudf::Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto result = g_vector_wilson_ctr_func(ctx, exp_cnt, clk_cnt);
    RUDF_DEBUG("{}", result.Size());
  }
}
BENCHMARK(BM_rapidudf_vector_wilson_ctr)->Setup(rapidudf_vector_wilson_ctr_setup);

BENCHMARK_MAIN();