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

#include <gtest/gtest.h>
#include <functional>
#include <vector>

#include "rapidudf/ast/context.h"
#include "rapidudf/ast/grammar.h"
#include "rapidudf/log/log.h"

using namespace rapidudf;
using namespace rapidudf::ast;

TEST(Grammar, simple) {
  std::string content = R"(
   //aaa
    int test_func(int a, float b){  //aaa
     //aaa
      if(a==1){return 1;}
      elif(a==2){return 2;}
      else { a=3;}
      int x = 0;
      while(x<3){
       a=a+1;
       a[x] = 1;
      }
      int y = f(2,3);
      a.b=1;
      test(2, a["a"][1].a, b.c.d);
      a["a"]=c;
     return (1+2);
    }
   
  )";
  //   std::string func = " float:bbb";

  //   int rc = parse_func_arg(func, f);
  //   ASSERT_EQ(rc, 0);
  ParseContext ctx;
  auto f = parse_function_ast(ctx, content);
  if (f.ok()) {
    RUDF_INFO("statements:{}", f->body.statements.size());
  }
}

TEST(Grammar, duplicate_var_name) {
  // spdlog::set_level(spdlog::level::debug);
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a, int a){  
     return 1;
    }
  )";

  auto f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());

  content = R"(
    int test_func(int a, int b){  
      int a = 1;
     return 1;
    }
  )";

  f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());
}

TEST(Grammar, var_not_exist) {
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a){
     c = 1;
     return 1;
    }
  )";

  auto f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());
}

TEST(Grammar, invalid_return) {
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a){
     return "hee";
    }
  )";
  auto f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());
}

TEST(Grammar, expression) {
  spdlog::set_level(spdlog::level::debug);
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a,int b, int c){
     return a*b+c;
    }
  )";
  auto f = parse_functions_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
}

TEST(Grammar, or_equals) {
  // spdlog::set_level(spdlog::level::debug);
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a){  
     return a||==1;
    }
  )";

  auto f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());
}

TEST(Grammar, array) {
  ParseContext ctx;
  std::string content = R"(
    int test_func(int a){  
     return a==[1,2,a];
    }
  )";
  auto f = parse_function_ast(ctx, content);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_FALSE(f.ok());
}

TEST(Grammar, rpn) {
  ParseContext ctx;
  std::string content = R"(
      2*(3+5)/4-8%2+4
  )";
  FunctionDesc desc;
  auto f = parse_expression_ast(ctx, content, desc);
  if (!f.ok()) {
    RUDF_ERROR("{}", f.status().ToString());
  }
  ASSERT_TRUE(f.ok());
  RUDF_ERROR("expr:{}", content);
  f->rpn_expr.Print();
}
