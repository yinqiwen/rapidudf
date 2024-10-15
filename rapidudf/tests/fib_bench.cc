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
#include <vector>

#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

static rapidudf::JitFunction<int, int> g_expr_func;

static void DoRapidUDFFibSetup(const benchmark::State& state) {
  std::string source = R"(
    int fib(int n) 
    { 
       if (n <= 1){
         return n; 
       }
       return fib(n - 1) + fib(n - 2); //递归调用
    } 
  )";
  rapidudf::JitCompiler compiler;
  auto result = compiler.CompileFunction<int, int>(source);
  g_expr_func = std::move(result.value());
}

static void DoRapidUDFFibTeardown(const benchmark::State& state) {}

static void BM_rapidudf_fib_func(benchmark::State& state) {
  uint64_t result = 0;
  size_t n = 0;
  for (auto _ : state) {
    result += g_expr_func(20);
    n++;
  }
}
BENCHMARK(BM_rapidudf_fib_func)->Setup(DoRapidUDFFibSetup)->Teardown(DoRapidUDFFibTeardown);

static void DoNativeFibSetup(const benchmark::State& state) {}

static void DooNativeFibTeardown(const benchmark::State& state) {}

static int __attribute__((noinline)) fib(int n) {
  if (n <= 1) {
    return n;
  }
  return fib(n - 1) + fib(n - 2);  // 递归调用
}

static void BM_native_fib_func(benchmark::State& state) {
  uint64_t result = 0;
  size_t n = 0;
  for (auto _ : state) {
    result += fib(20);
    n++;
  }
}
BENCHMARK(BM_native_fib_func)->Setup(DoNativeFibSetup)->Teardown(DooNativeFibTeardown);

BENCHMARK_MAIN();