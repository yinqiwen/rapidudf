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

#include <benchmark/benchmark.h>
#include <array>
#include <cmath>
#include <random>
#include <string_view>
#include <vector>

#include "rapidudf/functions/simd/string.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

#include "absl/strings/str_split.h"

constexpr std::string_view test_str =
    "v3:3950308951:学习分享:image,v3:1001599:英语:text,v3:108919712:英语学习:image,v3:207472163:英语短语:image,v3:"
    "205213337:英语语法:text,v3:70029117:顺口溜:text,v4:3950308951:学习分享:image,v4:1001599:英语:text,v4:108919712:"
    "英语学习:image,v4:207472163:英语短语:image,v4:205213337:英语语法:text,v4:70029117:顺口溜:text,v5:3950308951:"
    "学习分享:image,v5:1001599:英语:text,v5:108919712:英语学习:image,v5:207472163:英语短语:image,v5:205213337:英语语法:"
    "text,v5:70029117:顺口溜:text";

static void BM_rapidudf_simd_string_split(benchmark::State& state) {
  size_t n = 0;
  for (auto _ : state) {
    auto ss = rapidudf::functions::simd_string_split_by_char(test_str, ',');
    for (auto s : ss) {
      auto tag_list = rapidudf::functions::simd_string_split_by_char(s, ';');
      n += tag_list.size();
    }
  }
  RUDF_DEBUG("{}", n);
}
BENCHMARK(BM_rapidudf_simd_string_split);

static void BM_rapidudf_absl_string_split(benchmark::State& state) {
  size_t n = 0;
  for (auto _ : state) {
    std::vector<std::string_view> ss = absl::StrSplit(test_str, ',');
    for (auto s : ss) {
      std::vector<std::string_view> tag_list = absl::StrSplit(s, ';');
      n += tag_list.size();
    }
  }
  RUDF_DEBUG("{}", n);
}

BENCHMARK(BM_rapidudf_absl_string_split);

BENCHMARK_MAIN();