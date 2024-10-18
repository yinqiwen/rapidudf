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

#include <gtest/gtest.h>
#include <functional>
#include <stdexcept>
#include <vector>
#include "absl/strings/str_join.h"

#include "rapidudf/rapidudf.h"
#include "x86simdsort.h"

using namespace rapidudf;
using namespace rapidudf::ast;

TEST(JitCompiler, x86simdsort) {
  std::vector<float> data{1.1, 2.2, 0.1, 0.3, 0.2, 11, 12, 3, 423, 12, 12312, 12, 4124};
  auto result = x86simdsort::argsort(data.data(), data.size(), false, false);
  RUDF_INFO("result:[{}], data:[{}]", absl::StrJoin(result, ","), absl::StrJoin(data, ","));

  auto result1 = x86simdsort::argselect(data.data(), 2, data.size(), false);
  RUDF_INFO("result:[{}], data:[{}]", absl::StrJoin(result1, ","), absl::StrJoin(data, ","));

  size_t n = 16;
  std::vector<int> vals;
  std::vector<float> keys;
  for (size_t i = 0; i < n; i++) {
    keys.emplace_back(100.0 - 1.1 - i);
    vals.emplace_back(i);
  }
  x86simdsort::keyvalue_qsort(keys.data(), vals.data(), keys.size());
  RUDF_INFO("key:[{}], val:[{}]", absl::StrJoin(keys, ","), absl::StrJoin(vals, ","));
}

TEST(JitCompiler, sort) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<float> data{1.1, 2.2, 0.1, 0.3, 0.2, 11, 12, 3, 423, 12, 12312, 12, 4124};
  std::vector<float> data_clone = data;
  std::string content = R"(
   sort(x,true)
  )";
  Context ctx;
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<void, Context&, simd::Vector<float>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(ctx, data);
  std::sort(data_clone.begin(), data_clone.end(), std::greater<float>());
  for (size_t i = 0; i < data_clone.size(); i++) {
    ASSERT_FLOAT_EQ(data[i], data_clone[i]);
  }

  const std::vector<float>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}

TEST(JitCompiler, select) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 2067; i++) {
    data.emplace_back(1.1 + i % 800);
  }
  std::vector<double> data_clone = data;
  std::string content = R"(
   select(x,10,true)
  )";
  Context ctx;
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<void, Context&, simd::Vector<double>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(ctx, data);
  std::nth_element(data_clone.begin(), data_clone.begin() + 10, data_clone.end(), std::greater<double>());
  for (size_t i = 0; i < 10; i++) {
    RUDF_INFO("{} {}", data[i], data_clone[i]);
    // ASSERT_FLOAT_EQ(data[i], data_clone[i]);
  }
  const std::vector<double>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}

TEST(JitCompiler, topk) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 2067; i++) {
    data.emplace_back(1.1 + i % 800);
  }
  std::vector<double> data_clone = data;
  std::string content = R"(
   topk(x,10,true)
  )";
  Context ctx;
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<void, Context&, simd::Vector<double>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(ctx, data);
  for (size_t i = 0; i < 10; i++) {
    RUDF_INFO("{}", data[i]);
    // ASSERT_FLOAT_EQ(data[i], data_clone[i]);
  }
  const std::vector<double>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}

TEST(JitCompiler, argsort) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 16; i++) {
    data.emplace_back(1.1 + i % 800);
  }
  std::vector<double> data_clone = data;
  std::string content = R"(
   argsort(x,true)
  )";
  Context ctx;
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<simd::Vector<uint64_t>, Context&, simd::Vector<double>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, data);
  for (size_t i = 0; i < result.Size(); i++) {
    RUDF_INFO("{}", result[i]);
    // ASSERT_FLOAT_EQ(data[i], data_clone[i]);
  }
}

TEST(JitCompiler, argselect) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 2067; i++) {
    data.emplace_back(1.1 + i % 800);
  }
  std::vector<double> data_clone = data;
  std::string content = R"(
   argselect(x,16, false)
  )";
  Context ctx;
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<simd::Vector<uint64_t>, Context&, simd::Vector<double>>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto result = f(ctx, data);
  for (size_t i = 0; i < result.Size(); i++) {
    RUDF_INFO("[{}]:{}", i, result[i]);
    // ASSERT_FLOAT_EQ(data[i], data_clone[i]);
  }
}

TEST(JitCompiler, sort_kv) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 123; i++) {
    data.emplace_back(1.1 + i % 50);
  }

  Context ctx;
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<u32> test(Context ctx, simd_vector<f64> x){
        auto ids = iota(0_u32, x.size());
        sort_kv(x, ids,true);
        return ids;
    }
  )";

  auto rc = compiler.CompileFunction<simd::Vector<uint32_t>, Context&, simd::Vector<double>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto ids = f(ctx, data);
  for (size_t i = 0; i < ids.Size(); i++) {
    RUDF_INFO("[{}]:{}", ids[i], data[i]);
  }
  const std::vector<double>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}

TEST(JitCompiler, select_kv) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 123; i++) {
    data.emplace_back(1.1 + i % 50);
  }

  Context ctx;
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<u32> test(Context ctx, simd_vector<f64> x){
        auto ids = iota(0_u32, x.size());
        select_kv(x, ids, 10, false);
        return ids;
    }
  )";

  auto rc = compiler.CompileFunction<simd::Vector<uint32_t>, Context&, simd::Vector<double>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto ids = f(ctx, data);
  for (size_t i = 0; i < ids.Size(); i++) {
    RUDF_INFO("[{}]:{}", ids[i], data[i]);
  }
  const std::vector<double>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}

TEST(JitCompiler, topk_kv) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<double> data;
  for (size_t i = 0; i < 123; i++) {
    data.emplace_back(1.1 + i % 50);
  }

  Context ctx;
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<u32> test(Context ctx, simd_vector<f64> x){
        auto ids = iota(0_u32, x.size());
        topk_kv(x, ids, 10, true);
        return ids;
    }
  )";

  auto rc = compiler.CompileFunction<simd::Vector<uint32_t>, Context&, simd::Vector<double>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto ids = f(ctx, data);
  for (size_t i = 0; i < ids.Size(); i++) {
    RUDF_INFO("[{}]:{}", ids[i], data[i]);
  }
  const std::vector<double>& const_ref = data;
  ASSERT_THROW(f(ctx, const_ref), std::logic_error);
}