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
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
TEST(JitCompiler, while0) {
  JitCompiler compiler({.optimize_level = 0, .print_asm = true});
  std::string content = R"(
    int test_func(int x, int y){ 
      while(x > 0){
        y = y + 10;
        x = x-1;
      }
      return y;
    }
  )";
  auto rc = compiler.CompileFunction<int, int, int>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(10, 1), 101);
  ASSERT_EQ(f(100, 1), 1001);
}