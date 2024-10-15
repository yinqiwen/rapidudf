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

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

TEST(JitCompiler, vector_size) {
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  std::string content = R"(
    int test_f(int x){
       return x+1;
    }
    int test_func(int x){
      return test_f(x) + 10;
    }
  )";
  auto rc = compiler.CompileSource(content);
  ASSERT_TRUE(rc.ok());
  auto func_result = compiler.LoadFunction<int, int>("test_func");
  ASSERT_TRUE(func_result.ok());
  auto f = std::move(func_result.value());
  ASSERT_EQ(f(1), 12);
}