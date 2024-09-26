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
#include <unordered_map>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/simd_vector_table.h"

using namespace rapidudf;

static void print_column(simd::Column* c) {
  RUDF_INFO("print_column:{}", c->size());
  auto data = c->ToVector<double>().value();
  for (auto v : data) {
    RUDF_INFO("{}", v);
  }
}

RUDF_FUNC_REGISTER(print_column)

TEST(JitCompiler, table_simple) {
  Context ctx;
  simd::Table table(ctx);
  std::vector<double> d{1, 2, 3.1, 5.6};
  auto _ = table.Add("c0", std::move(d));
  std::string content = R"(
    print_column(x["c0"]+1.1)
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileExpression<void, Context&, simd::Table&>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());

  f(ctx, table);
}

TEST(JitCompiler, cond_table) {
  Context ctx;
  simd::Table table(ctx);
  std::vector<bool> d{true, false, false, true};
  auto _ = table.Add("c0", std::move(d));
  std::string content = R"(
    print_column(x["c0"]?1.1:2.2)
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileExpression<void, Context&, simd::Table&>(content, {"_", "x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(ctx, table);
}

TEST(JitCompiler, table_func) {
  Context ctx;
  simd::Table table(ctx);
  std::vector<double> d{1, 2, 3.1, 5.6};
  auto _ = table.Add("c0", std::move(d));
  std::string content = R"(
    void test(Context ctx, simd_table x){
      // var t =  x["c0"];
        x["c0"] += 3.3;
      //  print_column(x["c0"]);
       print_column(x["c0"]);
    }
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileFunction<void, Context&, simd::Table&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  f(ctx, table);
}

TEST(JitCompiler, table_func1) {
  Context ctx;
  simd::Table table(ctx);
  std::unordered_map<std::string, std::vector<double>> table_data;
  table_data["Click"] = {1, 2, 3, 5};
  table_data["Like"] = {1, 2, 3, 5};
  table_data["Join"] = {1, 2, 3, 5};
  table_data["Inter"] = {1, 2, 3, 5};
  table_data["TimeV1"] = {1, 2, 3, 5};
  table_data["PostComment"] = {1, 2, 3, 5};
  table_data["PositiveCommentV1"] = {1, 2, 3, 5};
  table_data["ExpoTimeV1"] = {1, 2, 3, 5};

  std::unordered_map<std::string, std::vector<double>> table_data_clone = table_data;
  // std::string multiple_pow =
  //     "(Click^10.0)*((Like+0.000082)^4.7)*(Inter^3.5)*((Join+0.000024)^5.5)*(TimeV1^7.0)*((PostComment+0.000024)^3.5)*("
  //     "(PositiveCommentV1+0.0038)^1.0)*(ExpoTimeV1^1.5)";
  auto _ = table.AddMap(std::move(table_data));
  std::string content = R"(
    (table["Click"]^10.0)*((table["Like"]+0.000082)^4.7)*(table["Inter"]^3.5)*((table["Join"]+0.000024)^5.5)*(table["TimeV1"]^7.0)*((table["PostComment"]+0.000024)^3.5)*((table["PositiveCommentV1"]+0.0038)^1.0)*(table["ExpoTimeV1"]^1.5)
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileExpression<simd::Column*, Context&, simd::Table&>(content, {"_", "table"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto column = f(ctx, table);
  simd::Vector<double> result = column->ToVector<double>().value();
  for (size_t i = 0; i < result.Size(); i++) {
    double actual =
        std::pow(table_data_clone["Click"][i], 10.0) * std::pow(table_data_clone["Like"][i] + 0.000082, 4.7) *
        std::pow(table_data_clone["Inter"][i], 3.5) * std::pow(table_data_clone["Join"][i] + 0.000024, 5.5) *
        std::pow(table_data_clone["TimeV1"][i], 7.0) * std::pow(table_data_clone["PostComment"][i] + 0.000024, 3.5) *
        std::pow(table_data_clone["PositiveCommentV1"][i] + 0.0038, 1.0) *
        std::pow(table_data_clone["ExpoTimeV1"][i], 1.5);
    ASSERT_DOUBLE_EQ(actual, result[i]);
  }
}