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