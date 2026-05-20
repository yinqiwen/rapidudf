/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
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

// rapidudf benchmark suite -------------------------------------------------
//
// Each scenario is benchmarked along the most relevant axes so that the
// numbers are directly comparable. The matrix is:
//
//   * scalar / native   : straight C++, function call per element
//   * scalar / jit      : RapidUDF JIT-compiled scalar function, called per element
//   * vector / native   : straight C++ loop over a buffer (compiler can auto-vectorize)
//   * vector / jit      : RapidUDF JIT-compiled SIMD function, one call processes the buffer
//
// Scenarios:
//   1. Trig expression  -- x + (cos(y - sin(2/x*pi)) - sin(x - cos(2*y/pi))) - y
//   2. Wilson CTR        -- log10/sqrt heavy formula
//   3. Order rule        -- branchy if/elif chain (scalar only)
//   4. Recursive Fibonacci -- recursion overhead
//
// All array-style scenarios use kBatchSize elements per benchmark iteration
// and report items/sec via `state.SetItemsProcessed`. Use --benchmark_filter
// to run a subset, e.g. `--benchmark_filter=expr` or `--benchmark_filter=jit`.

#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>

#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/string_view.h"

namespace {

constexpr size_t kBatchSize = 4099;
constexpr double kPi = 3.14159265358979323846264338327950288419716939937510;

using rapidudf::Context;
using rapidudf::JitCompiler;
using rapidudf::JitFunction;
using rapidudf::Vector;

// ---------------------------------------------------------------------------
// Common helper: compile a JIT function once or die. Used in Setup callbacks.
// ---------------------------------------------------------------------------
template <typename Ret, typename... Args>
JitFunction<Ret, Args...> CompileOrDie(const std::string& source) {
  JitCompiler compiler;
  auto result = compiler.CompileFunction<Ret, Args...>(source);
  if (!result.ok()) {
    RUDF_ERROR("JIT compile failed: {}", result.status().ToString());
    std::abort();
  }
  return std::move(result.value());
}

// ===========================================================================
// SCENARIO 1 -- Trig expression
//   f(x, y) = x + (cos(y - sin(2/x*pi)) - sin(x - cos(2*y/pi))) - y
// ===========================================================================

namespace expr {

std::vector<double> g_x;
std::vector<double> g_y;
JitFunction<double, double, double, double> g_jit_scalar;
JitFunction<Vector<double>, Context&, Vector<double>, Vector<double>> g_jit_vector;

void InitInputs() {
  g_x.clear();
  g_y.clear();
  g_x.reserve(kBatchSize);
  g_y.reserve(kBatchSize);
  for (size_t i = 0; i < kBatchSize; ++i) {
    g_x.push_back(static_cast<double>(i + 1));
    g_y.push_back(static_cast<double>(i + 101));
  }
}

inline double NativeKernel(double x, double y) {
  return x + (std::cos(y - std::sin(2.0 / x * kPi)) - std::sin(x - std::cos(2.0 * y / kPi))) - y;
}

void SetupScalar(const benchmark::State&) {
  InitInputs();
  g_jit_scalar = CompileOrDie<double, double, double, double>(R"(
    double expr_scalar(double x, double y, double pi) {
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )");
}

void SetupVector(const benchmark::State&) {
  InitInputs();
  g_jit_vector =
      CompileOrDie<Vector<double>, Context&, Vector<double>, Vector<double>>(R"(
    simd_vector<f64> expr_vector(Context ctx, simd_vector<f64> x, simd_vector<f64> y) {
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )");
}

void BM_expr_scalar_native(benchmark::State& state) {
  for (auto _ : state) {
    double sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      sink += NativeKernel(g_x[i], g_y[i]);
    }
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

void BM_expr_scalar_jit(benchmark::State& state) {
  for (auto _ : state) {
    double sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      sink += g_jit_scalar(g_x[i], g_y[i], kPi);
    }
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

// Vector / native baseline: a single inline loop over the whole batch.
// The compiler is free to auto-vectorize at -O2/-O3.
void __attribute__((noinline))
NativeKernelVector(const double* x, const double* y, double* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = NativeKernel(x[i], y[i]);
  }
}

void BM_expr_vector_native(benchmark::State& state) {
  std::vector<double> out(kBatchSize);
  for (auto _ : state) {
    NativeKernelVector(g_x.data(), g_y.data(), out.data(), kBatchSize);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

void BM_expr_vector_jit(benchmark::State& state) {
  Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto out = g_jit_vector(ctx, g_x, g_y);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

}  // namespace expr

BENCHMARK(expr::BM_expr_scalar_native)->Setup(expr::SetupScalar);
BENCHMARK(expr::BM_expr_scalar_jit)->Setup(expr::SetupScalar);
BENCHMARK(expr::BM_expr_vector_native)->Setup(expr::SetupScalar);
BENCHMARK(expr::BM_expr_vector_jit)->Setup(expr::SetupVector);

// ===========================================================================
// SCENARIO 2 -- Wilson CTR
//   ctr(exp, clk) = log10(exp) * (... clk/exp ... sqrt ...) / (...)
//
// Benchmarked with both f64 and f32 element types so the SIMD speedup at
// narrower lane widths is visible.
// ===========================================================================

namespace wilson {

// Shared inline kernel parameterised on element type. T(1.96) etc. avoids
// any unwanted f64 promotion during the float computation.
template <typename T>
inline T InlineKernel(T exp_cnt, T clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + T(1.96) * T(1.96) / (T(2) * exp_cnt) -
          T(1.96) / (T(2) * exp_cnt) *
              std::sqrt(T(4) * exp_cnt * (T(1) - clk_cnt / exp_cnt) * clk_cnt / exp_cnt +
                        T(1.96) * T(1.96))) /
         (T(1) + T(1.96) * T(1.96) / exp_cnt);
}

// noinline so each call is a real function call -- prevents the outer
// benchmark loop from being auto-vectorized over the scalar baseline.
double __attribute__((noinline)) NativeScalarF64(double e, double c) {
  return InlineKernel<double>(e, c);
}
float __attribute__((noinline)) NativeScalarF32(float e, float c) {
  return InlineKernel<float>(e, c);
}

// Inlinable buffer-loop kernels: compiler is free to auto-vectorise.
void __attribute__((noinline))
NativeVectorF64(const double* exp_cnt, const double* clk_cnt, double* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = InlineKernel<double>(exp_cnt[i], clk_cnt[i]);
}
void __attribute__((noinline))
NativeVectorF32(const float* exp_cnt, const float* clk_cnt, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = InlineKernel<float>(exp_cnt[i], clk_cnt[i]);
}

// Inputs (same logical values, materialised as both f32 and f64).
std::vector<double> g_exp_f64;
std::vector<double> g_clk_f64;
std::vector<float> g_exp_f32;
std::vector<float> g_clk_f32;

void InitInputs() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distr(1, 100);
  g_exp_f64.clear();
  g_clk_f64.clear();
  g_exp_f32.clear();
  g_clk_f32.clear();
  g_exp_f64.reserve(kBatchSize);
  g_clk_f64.reserve(kBatchSize);
  g_exp_f32.reserve(kBatchSize);
  g_clk_f32.reserve(kBatchSize);
  for (size_t i = 0; i < kBatchSize; ++i) {
    int clk = distr(gen);
    int exp = clk + 10;
    g_clk_f64.push_back(static_cast<double>(clk));
    g_exp_f64.push_back(static_cast<double>(exp));
    g_clk_f32.push_back(static_cast<float>(clk));
    g_exp_f32.push_back(static_cast<float>(exp));
  }
}

// JIT functions per element type.
JitFunction<double, double, double> g_jit_scalar_f64;
JitFunction<float, float, float> g_jit_scalar_f32;
JitFunction<Vector<double>, Context&, Vector<double>, Vector<double>> g_jit_vector_f64;
JitFunction<Vector<float>, Context&, Vector<float>, Vector<float>> g_jit_vector_f32;

constexpr const char* kScalarSourceF64 = R"(
  f64 wilson_ctr(f64 exp_cnt, f64 clk_cnt) {
    return log10(exp_cnt) *
           (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
            1.96 / (2 * exp_cnt) *
              sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
           (1 + 1.96 * 1.96 / exp_cnt);
  }
)";

constexpr const char* kScalarSourceF32 = R"(
  f32 wilson_ctr(f32 exp_cnt, f32 clk_cnt) {
    return log10(exp_cnt) *
           (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
            1.96 / (2 * exp_cnt) *
              sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
           (1 + 1.96 * 1.96 / exp_cnt);
  }
)";

constexpr const char* kVectorSourceF64 = R"(
  simd_vector<f64> wilson_ctr(Context ctx, simd_vector<f64> exp_cnt, simd_vector<f64> clk_cnt) {
    return log10(exp_cnt) *
           (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
            1.96 / (2 * exp_cnt) *
              sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
           (1 + 1.96 * 1.96 / exp_cnt);
  }
)";

constexpr const char* kVectorSourceF32 = R"(
  simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> exp_cnt, simd_vector<f32> clk_cnt) {
    return log10(exp_cnt) *
           (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
            1.96 / (2 * exp_cnt) *
              sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
           (1 + 1.96 * 1.96 / exp_cnt);
  }
)";

void SetupF64Scalar(const benchmark::State&) {
  InitInputs();
  g_jit_scalar_f64 = CompileOrDie<double, double, double>(kScalarSourceF64);
}
void SetupF64Vector(const benchmark::State&) {
  InitInputs();
  g_jit_vector_f64 =
      CompileOrDie<Vector<double>, Context&, Vector<double>, Vector<double>>(kVectorSourceF64);
}
void SetupF32Scalar(const benchmark::State&) {
  InitInputs();
  g_jit_scalar_f32 = CompileOrDie<float, float, float>(kScalarSourceF32);
}
void SetupF32Vector(const benchmark::State&) {
  InitInputs();
  g_jit_vector_f32 =
      CompileOrDie<Vector<float>, Context&, Vector<float>, Vector<float>>(kVectorSourceF32);
}

// f64 ----------------------------------------------------------------------

void BM_wilson_ctr_f64_scalar_native(benchmark::State& state) {
  for (auto _ : state) {
    double sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) sink += NativeScalarF64(g_exp_f64[i], g_clk_f64[i]);
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f64_scalar_jit(benchmark::State& state) {
  for (auto _ : state) {
    double sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) sink += g_jit_scalar_f64(g_exp_f64[i], g_clk_f64[i]);
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f64_vector_native(benchmark::State& state) {
  std::vector<double> out(kBatchSize);
  for (auto _ : state) {
    NativeVectorF64(g_exp_f64.data(), g_clk_f64.data(), out.data(), kBatchSize);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f64_vector_jit(benchmark::State& state) {
  Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto out = g_jit_vector_f64(ctx, g_exp_f64, g_clk_f64);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

// f32 ----------------------------------------------------------------------

void BM_wilson_ctr_f32_scalar_native(benchmark::State& state) {
  for (auto _ : state) {
    float sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) sink += NativeScalarF32(g_exp_f32[i], g_clk_f32[i]);
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f32_scalar_jit(benchmark::State& state) {
  for (auto _ : state) {
    float sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) sink += g_jit_scalar_f32(g_exp_f32[i], g_clk_f32[i]);
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f32_vector_native(benchmark::State& state) {
  std::vector<float> out(kBatchSize);
  for (auto _ : state) {
    NativeVectorF32(g_exp_f32.data(), g_clk_f32.data(), out.data(), kBatchSize);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
void BM_wilson_ctr_f32_vector_jit(benchmark::State& state) {
  Context ctx;
  for (auto _ : state) {
    ctx.Reset();
    auto out = g_jit_vector_f32(ctx, g_exp_f32, g_clk_f32);
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

}  // namespace wilson

BENCHMARK(wilson::BM_wilson_ctr_f64_scalar_native)->Setup(wilson::SetupF64Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f64_scalar_jit)->Setup(wilson::SetupF64Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f64_vector_native)->Setup(wilson::SetupF64Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f64_vector_jit)->Setup(wilson::SetupF64Vector);
BENCHMARK(wilson::BM_wilson_ctr_f32_scalar_native)->Setup(wilson::SetupF32Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f32_scalar_jit)->Setup(wilson::SetupF32Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f32_vector_native)->Setup(wilson::SetupF32Scalar);
BENCHMARK(wilson::BM_wilson_ctr_f32_vector_jit)->Setup(wilson::SetupF32Vector);

// ===========================================================================
// SCENARIO 3 -- Branchy order-rule classification (scalar only)
// ===========================================================================
// Order must live at file/global scope so RUDF_STRUCT_FIELDS registers it
// under the bare name `Order` that the JIT source references.

}  // namespace
struct Order {
  int amount = 0;
};
RUDF_STRUCT_FIELDS(Order, amount)
namespace {

namespace order_rule {

std::vector<Order> g_orders;
JitFunction<int, Order&> g_jit;

void InitInputs() {
  g_orders.clear();
  g_orders.reserve(kBatchSize);
  // Spread amounts across all four buckets (0, 100, 500, 1000).
  for (size_t i = 0; i < kBatchSize; ++i) {
    g_orders.push_back(Order{static_cast<int>(i % 1500)});
  }
}

int __attribute__((noinline)) NativeRule(int amount) {
  if (amount < 100) {
    return 0;
  } else if (amount < 500) {
    return 100;
  } else if (amount < 1000) {
    return 500;
  } else {
    return 1000;
  }
}

void Setup(const benchmark::State&) {
  InitInputs();
  g_jit = CompileOrDie<int, Order&>(R"(
    int rule_func(Order order) {
      if (order.amount < 100) {
        return 0;
      } elif (order.amount >= 100 && order.amount < 500) {
        return 100;
      } elif (order.amount >= 500 && order.amount < 1000) {
        return 500;
      } else {
        return 1000;
      }
    }
  )");
}

void BM_order_rule_native(benchmark::State& state) {
  for (auto _ : state) {
    int sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      sink += NativeRule(g_orders[i].amount);
    }
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

void BM_order_rule_jit(benchmark::State& state) {
  for (auto _ : state) {
    int sink = 0;
    for (size_t i = 0; i < kBatchSize; ++i) {
      sink += g_jit(g_orders[i]);
    }
    benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}

}  // namespace order_rule

BENCHMARK(order_rule::BM_order_rule_native)->Setup(order_rule::Setup);
BENCHMARK(order_rule::BM_order_rule_jit)->Setup(order_rule::Setup);

// ===========================================================================
// SCENARIO 4 -- Recursive Fibonacci (scalar only)
// ===========================================================================

namespace fib {

constexpr int kFibN = 20;
JitFunction<int, int> g_jit;

int __attribute__((noinline)) NativeFib(int n) {
  if (n <= 1) return n;
  return NativeFib(n - 1) + NativeFib(n - 2);
}

void Setup(const benchmark::State&) {
  g_jit = CompileOrDie<int, int>(R"(
    int fib(int n) {
      if (n <= 1) {
        return n;
      }
      return fib(n - 1) + fib(n - 2);
    }
  )");
}

void BM_fib_native(benchmark::State& state) {
  int sink = 0;
  for (auto _ : state) {
    sink += NativeFib(kFibN);
    benchmark::DoNotOptimize(sink);
  }
}

void BM_fib_jit(benchmark::State& state) {
  int sink = 0;
  for (auto _ : state) {
    sink += g_jit(kFibN);
    benchmark::DoNotOptimize(sink);
  }
}

}  // namespace fib

BENCHMARK(fib::BM_fib_native);
BENCHMARK(fib::BM_fib_jit)->Setup(fib::Setup);

}  // namespace

BENCHMARK_MAIN();
