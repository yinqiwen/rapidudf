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
#include <vector>
#include "rapidudf/rapidudf.h"
using namespace rapidudf;

TEST(JitCompiler, ifelse0) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;

  std::string content = R"(
    int test_func(int x){ 
      if(x>10){
         return 20;
      }elif(x > 5){
       return 10;
      }else{
        return 0;
      }
    }
  )";
  auto rc = compiler.CompileFunction<int, int>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_DOUBLE_EQ(f(11), 20);
  ASSERT_DOUBLE_EQ(f(9), 10);
  ASSERT_DOUBLE_EQ(f(6), 10);
  ASSERT_DOUBLE_EQ(f(4), 0);
}

TEST(JitCompiler, ifelse1) {
  spdlog::set_level(spdlog::level::debug);
  JitCompiler compiler;

  std::string content = R"(
    int test_func(int x){ 
      if(x>10){
         x=20;
      }elif(x > 5){
       x= 10;
      }else{
        x= 0;
      }
      return x;
    }
  )";
  auto rc = compiler.CompileFunction<int, int>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_DOUBLE_EQ(f(11), 20);
  ASSERT_DOUBLE_EQ(f(9), 10);
  ASSERT_DOUBLE_EQ(f(6), 10);
  ASSERT_DOUBLE_EQ(f(4), 0);
}