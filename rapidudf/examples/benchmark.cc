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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "rapidudf/rapidudf.h"

using namespace rapidudf;

constexpr std::string_view global_expression_list[] = {
    "(y + x)",
    "2 * (y + x)",
    "(2 * y + 2 * x)",
    "((1.23 * x^2) / y) - 123.123",
    "(y + x / y) * (x - y / x)",
    "x / ((x + y) + (x - y)) / y",
    "1 - ((x * y) + (y / x)) - 3",
    "(5.5 + x) + (2 * x - 2 / 3 * y) * (x / 3 + y / 4) + (y + 7.7)",
    "1.1*x^1 + 2.2*y^2 - 3.3*x^3 + 4.4*y^15 - 5.5*x^23 + 6.6*y^55",
    "sin(2 * x) + cos(pi / y)",
    "1 - sin(2 * x) + cos(pi / y)",
    "sqrt(111.111 - sin(2 * x) + cos(pi / y) / 333.333)",
    "(x^2 / sin(2 * pi / y)) - x / 2",
    "x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y",
    "clamp(-1.0, sin(2 * pi * x) + cos(y / 2 * pi), +1.0)",
    "max(3.33, min(sqrt(1 - sin(2 * x) + cos(pi / y) / 3), 1.11))",
    "((y + (x * 2.2)) <= (x + y + 1.1))?(x - y):(x * y) + 2 * pi / x"};
constexpr std::size_t global_expression_list_size = sizeof(global_expression_list) / sizeof(std::string_view);
static const double pi = 3.141592653589793238462643383279502;
static const double global_lower_bound_x = -100.0;
static const double global_lower_bound_y = -100.0;
static const double global_upper_bound_x = +100.0;
static const double global_upper_bound_y = +100.0;
static const double global_delta = 0.0111;

static std::vector<JitFunction<double, double, double>> rapidudf_expr_funcs;

template <typename T>
struct native {
  //   typedef typename exprtk::details::functor_t<T> functor_t;
  //   typedef typename functor_t::Type Type;
  using Type = T;
  static inline T avg(Type x, Type y) { return (x + y) / T(2); }

  static inline T clamp(const Type l, const Type v, const Type u) { return ((v < l) ? l : ((v > u) ? u : v)); }

  static inline T func00(Type x, Type y) { return (y + x); }

  static inline T func01(Type x, Type y) { return T(2) * (y + x); }

  static inline T func02(Type x, Type y) { return (T(2) * y + T(2) * x); }

  static inline T func03(Type x, Type y) { return ((T(1.23) * (x * x)) / y) - T(123.123); }

  static inline T func04(Type x, Type y) { return (y + x / y) * (x - y / x); }

  static inline T func05(Type x, Type y) { return x / ((x + y) + (x - y)) / y; }

  static inline T func06(Type x, Type y) { return T(1) - ((x * y) + (y / x)) - T(3); }

  static inline T func07(Type x, Type y) {
    return (T(5.5) + x) + (T(2) * x - T(2) / T(3) * y) * (x / T(3) + y / T(4)) + (y + T(7.7));
  }

  static inline T func08(Type x, Type y) {
    using namespace std;
    return (T(1.1) * pow(x, T(1)) + T(2.2) * pow(y, T(2)) - T(3.3) * pow(x, T(3)) + T(4.4) * pow(y, T(15)) -
            T(5.5) * pow(x, T(23)) + T(6.6) * pow(y, T(55)));
  }

  static inline T func09(Type x, Type y) { return std::sin(T(2) * x) + std::cos(pi / y); }

  static inline T func10(Type x, Type y) { return T(1) - std::sin(T(2) * x) + std::cos(pi / y); }

  static inline T func11(Type x, Type y) {
    return std::sqrt(T(111.111) - std::sin(T(2) * x) + std::cos(pi / y) / T(333.333));
  }

  static inline T func12(Type x, Type y) { return ((x * x) / std::sin(T(2) * pi / y)) - x / T(2); }

  static inline T func13(Type x, Type y) {
    return (x + (std::cos(y - std::sin(T(2) / x * pi)) - std::sin(x - std::cos(T(2) * y / pi))) - y);
  }

  static inline T func14(Type x, Type y) {
    return clamp(T(-1), std::sin(T(2) * pi * x) + std::cos(y / T(2) * pi), +T(1));
  }

  static inline T func15(Type x, Type y) {
    return std::max(T(3.33), std::min(sqrt(T(1) - std::sin(T(2) * x) + std::cos(pi / y) / T(3)), T(1.11)));
  }

  static inline T func16(Type x, Type y) {
    return (((y + (x * T(2.2))) <= (x + y + T(1.1))) ? x - y : x * y) + T(2) * pi / x;
  }
};

template <typename T>
void run_rapidudf_benchmark(T& x, T& y, JitFunction<T, T, T>& f, std::string_view expr_string) {
  T total = T(0);
  unsigned int count = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (x = global_lower_bound_x; x <= global_upper_bound_x; x += global_delta) {
    for (y = global_lower_bound_y; y <= global_upper_bound_y; y += global_delta) {
      total += f(x, y);
      ++count;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (T(0) != total) {
    double secs = duration.count() / 1000000.0;
    RUDF_INFO("[rapidudf] Total Time:{}s  Rate:{}evals/sec Expression:{}", secs, count / secs, expr_string);
  } else {
    RUDF_ERROR("run_rapidudf_benchmark() - Error running benchmark for expression:", expr_string);
  }
}

template <typename T, typename NativeFunction>
void run_native_benchmark(T& x, T& y, NativeFunction f, std::string_view expr_string) {
  T total = T(0);
  unsigned int count = 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (x = global_lower_bound_x; x <= global_upper_bound_x; x += global_delta) {
    for (y = global_lower_bound_y; y <= global_upper_bound_y; y += global_delta) {
      total += f(x, y);
      ++count;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  if (T(0) != total) {
    double secs = duration.count() / 1000000.0;
    RUDF_INFO("[native] Total Time:{}s  Rate:{}evals/sec Expression:{}", secs, count / secs, expr_string);
  } else {
    RUDF_ERROR("run_native_benchmark() - Error running benchmark for expression:", expr_string);
  }
}

double pgo_primer() {
  static const double lower_bound_x = -50.0;
  static const double lower_bound_y = -50.0;
  static const double upper_bound_x = +50.0;
  static const double upper_bound_y = +50.0;
  static const double delta = 0.07;

  double total = 0.0;

  for (double x = lower_bound_x; x <= upper_bound_x; x += delta) {
    for (double y = lower_bound_y; y <= upper_bound_y; y += delta) {
      total += native<double>::func00(x, y);
      total += native<double>::func01(x, y);
      total += native<double>::func02(x, y);
      total += native<double>::func03(x, y);
      total += native<double>::func04(x, y);
      total += native<double>::func05(x, y);
      total += native<double>::func06(x, y);
      total += native<double>::func07(x, y);
      total += native<double>::func08(x, y);
      total += native<double>::func09(x, y);
      total += native<double>::func10(x, y);
      total += native<double>::func11(x, y);
      total += native<double>::func12(x, y);
      total += native<double>::func13(x, y);
      total += native<double>::func14(x, y);
      total += native<double>::func15(x, y);
      total += native<double>::func16(x, y);
    }
  }

  return total;
}

template <typename T>
static bool run_parse_benchmark() {
  static const std::size_t rounds = 100;

  for (std::size_t i = 0; i < global_expression_list_size; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();

    JitFunction<T, T, T> jit_func;
    for (std::size_t r = 0; r < rounds; ++r) {
      JitCompiler compiler;
      auto result = compiler.CompileExpression<T, T, T>(std::string(global_expression_list[i]), {"x", "y"});
      if (!result.ok()) {
        RUDF_ERROR("Failed to compile {} with error:{}", global_expression_list[i], result.status().ToString());
        exit(1);
      }
      jit_func = std::move(result.value());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double secs = duration.count() / 1000000.0;
    RUDF_INFO("[parse] Total Time:{}s  Rate:{}parse/sec Expression: {}", secs, rounds / secs,
              global_expression_list[i]);
  }

  return true;
}

static void init_rapidudf_expr_funcs() {
  for (std::size_t i = 0; i < global_expression_list_size; ++i) {
    JitCompiler compiler;
    auto result =
        compiler.CompileExpression<double, double, double>(std::string(global_expression_list[i]), {"x", "y"});
    if (!result.ok()) {
      RUDF_ERROR("Failed to compile {} with error:{}", global_expression_list[i], result.status().ToString());
      exit(1);
    }
    auto f = std::move(result.value());
    rapidudf_expr_funcs.emplace_back(std::move(f));
  }
}

int main(int argc, char* argv[]) {
  // if (argc >= 2) {
  //   const std::string file_name = argv[1];

  //   if (argc == 2)
  //     perform_file_based_benchmark(file_name);
  //   else
  //     perform_file_based_benchmark(file_name, atoi(argv[2]));

  //   return 0;
  // }

  pgo_primer();
  init_rapidudf_expr_funcs();

  double x = 0;
  double y = 0;

  // exprtk::symbol_table<double> symbol_table;
  // symbol_table.add_constants();
  // symbol_table.add_variable("x", x);
  // symbol_table.add_variable("y", y);

  // std::deque<exprtk::expression<double> > compiled_expr_list;

  // if (!load_expression(symbol_table, compiled_expr_list)) {
  //   return 1;
  // }
  {
    std::cout << "--- RapidUDF Parse---" << std::endl;
    run_parse_benchmark<double>();
  }
  {
    std::cout << "--- RapidUDF ---" << std::endl;
    for (std::size_t i = 0; i < rapidudf_expr_funcs.size(); ++i) {
      run_rapidudf_benchmark(x, y, rapidudf_expr_funcs[i], global_expression_list[i]);
    }
  }

  {
    std::cout << "--- NATIVE ---" << std::endl;
    run_native_benchmark(x, y, native<double>::func00, global_expression_list[0]);
    run_native_benchmark(x, y, native<double>::func01, global_expression_list[1]);
    run_native_benchmark(x, y, native<double>::func02, global_expression_list[2]);
    run_native_benchmark(x, y, native<double>::func03, global_expression_list[3]);
    run_native_benchmark(x, y, native<double>::func04, global_expression_list[4]);
    run_native_benchmark(x, y, native<double>::func05, global_expression_list[5]);
    run_native_benchmark(x, y, native<double>::func06, global_expression_list[6]);
    run_native_benchmark(x, y, native<double>::func07, global_expression_list[7]);
    run_native_benchmark(x, y, native<double>::func08, global_expression_list[8]);
    run_native_benchmark(x, y, native<double>::func09, global_expression_list[9]);
    run_native_benchmark(x, y, native<double>::func10, global_expression_list[10]);
    run_native_benchmark(x, y, native<double>::func11, global_expression_list[11]);
    run_native_benchmark(x, y, native<double>::func12, global_expression_list[12]);
    run_native_benchmark(x, y, native<double>::func13, global_expression_list[13]);
    run_native_benchmark(x, y, native<double>::func14, global_expression_list[14]);
    run_native_benchmark(x, y, native<double>::func15, global_expression_list[15]);
    run_native_benchmark(x, y, native<double>::func16, global_expression_list[16]);
  }
  return 0;
}
