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
// #include "exprtk.hpp"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

static std::vector<rapidudf::JitFunction<double, double, double, double>> g_expr_funcs;
static const double pi = 3.14159265358979323846264338327950288419716939937510;
static void DoRapidUDFExprSetup(const benchmark::State& state) {
  // std::string expr = "x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y";
  std::string source = R"(
    double test_func(double x,double y, double pi){
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )";
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<double, double, double, double>(source, false);
  g_expr_funcs.emplace_back(std::move(result.value()));
}

static void DoRapidUDFExprTeardown(const benchmark::State& state) {}

static void BM_rapidudf_expr_func(benchmark::State& state) {
  double result = 0;
  size_t n = 0;
  for (auto _ : state) {
    result += g_expr_funcs[0](100, 102, pi);
    n++;
  }
  double x = 100;
  double y = 102;
  double actual = x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
  RUDF_INFO("Result:{} for loop:{}, actual:{}", result, n, actual);
}
BENCHMARK(BM_rapidudf_expr_func)->Setup(DoRapidUDFExprSetup)->Teardown(DoRapidUDFExprTeardown);

// static exprtk::parser<double> exprtk_parser;
// static exprtk::expression<double> exprtk_expression;
// static exprtk::symbol_table<double> exprtk_symbol_table;
// static double exprtk_x = 100;
// static double exprtk_y = 102;
// static void DoExprtkExprSetup(const benchmark::State& state) {
//   exprtk_symbol_table.add_constants();
//   exprtk_symbol_table.add_variable("x", exprtk_x);
//   exprtk_symbol_table.add_variable("y", exprtk_y);
//   exprtk_expression.register_symbol_table(exprtk_symbol_table);
//   std::string expr = "x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y";
//   // std::string expr = "cos(y - sin(2 / x * pi))";
//   if (!exprtk_parser.compile(expr, exprtk_expression)) {
//     RUDF_ERROR("[load_expression] - Parser Error:{}\tExpression: {}", exprtk_parser.error().c_str(), expr);
//   }
// }

// static void DooExprtkExprTeardown(const benchmark::State& state) {}

// static double __attribute__((noinline)) invoke_exprtk() { return exprtk_expression.value(); }
// static void BM_exprtk_expr_func(benchmark::State& state) {
//   double total = 0;
//   size_t n = 0;
//   for (auto _ : state) {
//     total += invoke_exprtk();
//     n++;
//   }
//   RUDF_INFO("Result:{} for loop:{}", total, n);
// }
// BENCHMARK(BM_exprtk_expr_func)->Setup(DoExprtkExprSetup)->Teardown(DooExprtkExprTeardown);

BENCHMARK_MAIN();