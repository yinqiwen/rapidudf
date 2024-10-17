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
#include <array>
#include <cmath>
#include <random>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/simd/vector.h"

static std::random_device rd;
static std::mt19937 gen(rd());
using Embedding = std::array<float, 64>;
static std::vector<Embedding> left, right;
static size_t left_n = 500;
static size_t right_n = 300;
static std::vector<std::vector<float>> results;

static void rapidudf_vector_dot_setup(const benchmark::State& state) {
  left.clear();
  right.clear();
  results.clear();
  std::uniform_int_distribution<> distr(1, 100);
  for (size_t i = 0; i < left_n; i++) {
    Embedding embedding;
    for (size_t j = 0; j < embedding.size(); j++) {
      embedding[j] = static_cast<float>(distr(gen));
    }
    left.emplace_back(std::move(embedding));
  }
  for (size_t i = 0; i < right_n; i++) {
    Embedding embedding;
    for (size_t j = 0; j < embedding.size(); j++) {
      embedding[j] = static_cast<float>(distr(gen));
    }
    right.emplace_back(std::move(embedding));
  }
}

static void BM_rapidudf_vector_dot(benchmark::State& state) {
  for (auto _ : state) {
    results.clear();
    results.reserve(left.size());
    for (auto& left_embedding : left) {
      std::vector<float> result;
      result.reserve(right.size());
      rapidudf::simd::Vector<float> left_operand(left_embedding.data(), left_embedding.size());
      for (auto& right_embedding : right) {
        rapidudf::simd::Vector<float> right_operand(right_embedding.data(), right_embedding.size());
        result.emplace_back(rapidudf::functions::simd_vector_dot(left_operand, right_operand));
      }
      results.emplace_back(std::move(result));
    }
    RUDF_DEBUG("{}", results.size());
  }
}
BENCHMARK(BM_rapidudf_vector_dot)->Setup(rapidudf_vector_dot_setup);

BENCHMARK_MAIN();