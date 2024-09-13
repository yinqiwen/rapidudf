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