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
#include <tuple>
#include <unordered_map>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/vector/table.h"
#include "rapidudf/vector/vector.h"

using namespace rapidudf;

TEST(JitCompiler, table_simple) {
  auto schema = simd::TableSchema::GetOrCreate("mytable", [](simd::TableSchema* s) {
    std::ignore = s->AddColumn<StringView>("sv");
    std::ignore = s->AddColumn<float>("fv");
    std::ignore = s->AddColumn<int>("iv");
    std::ignore = s->AddColumn<bool>("bv");
  });
  Context ctx;
  auto obj = schema->NewTable(ctx);
  std::vector<float> fv{1, 2, 3.1, 5.6};
  std::vector<int> iv{10, 20, 30, 540};
  std::vector<std::string> sv{"a0", "a1", "a2", "a3"};
  std::vector<bool> bv{false, true, false, true, false};
  std::ignore = obj->Set("fv", fv);
  std::ignore = obj->Set("iv", iv);
  std::ignore = obj->Set("sv", sv);
  std::ignore = obj->Set("bv", bv);

  auto fv_result = obj->Get<float>("fv");
  ASSERT_TRUE(fv_result.ok());
  ASSERT_EQ(fv_result.value().Size(), fv.size());
  for (size_t i = 0; i < fv.size(); i++) {
    ASSERT_FLOAT_EQ(fv_result.value()[i], fv[i]);
  }

  auto sv_result = obj->Get<StringView>("sv");
  ASSERT_TRUE(sv_result.ok());
  ASSERT_EQ(sv_result.value().Size(), sv.size());
  ASSERT_TRUE(sv_result.value().IsReadonly());
  for (size_t i = 0; i < sv.size(); i++) {
    ASSERT_EQ(sv_result.value()[i], StringView(sv[i]));
  }

  auto bv_result = obj->Get<bool>("bv");
  ASSERT_TRUE(bv_result.ok());
  ASSERT_EQ(bv_result.value().Size(), bv.size());
  for (size_t i = 0; i < bv.size(); i++) {
    ASSERT_EQ(bv_result.value()[i], bv[i]);
  }
}

TEST(JitCompiler, table_func1) {
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

  auto schema = simd::TableSchema::GetOrCreate("score_table", [&](simd::TableSchema* s) {
    for (auto& [name, _] : table_data) {
      std::ignore = s->AddColumn<double>(name);
    }
  });
  Context ctx;
  auto table = schema->NewTable(ctx);
  auto _ = table->AddMap(std::move(table_data));
  // auto _ = table.AddMap(std::move(table_data));

  std::string multiple_pow =
      "(Click^10.0)*((Like+0.000082)^4.7)*(Inter^3.5)*((Join+0.000024)^5.5)*(TimeV1^7.0)*((PostComment+0.000024)^3.5)*("
      "(PositiveCommentV1+0.0038)^1.0)*(ExpoTimeV1^1.5)";

  std::string content = R"(
    (table.Click^10.0)*((table.Like+0.000082)^4.7)*(table.Inter^3.5)*((table.Join+0.000024)^5.5)*(table.TimeV1^7.0)*((table.PostComment+0.000024)^3.5)*((table.PositiveCommentV1+0.0038)^1.0)*(table.ExpoTimeV1^1.5)
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<simd::Vector<double>, Context&, simd::Table&>(
      content, {{"_"}, {"table", "score_table"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto column = f(ctx, *table);
  //   simd::Vector<double> result = column->ToVector<double>().value();
  for (size_t i = 0; i < column.Size(); i++) {
    double actual =
        std::pow(table_data_clone["Click"][i], 10.0) * std::pow(table_data_clone["Like"][i] + 0.000082, 4.7) *
        std::pow(table_data_clone["Inter"][i], 3.5) * std::pow(table_data_clone["Join"][i] + 0.000024, 5.5) *
        std::pow(table_data_clone["TimeV1"][i], 7.0) * std::pow(table_data_clone["PostComment"][i] + 0.000024, 3.5) *
        std::pow(table_data_clone["PositiveCommentV1"][i] + 0.0038, 1.0) *
        std::pow(table_data_clone["ExpoTimeV1"][i], 1.5);
    ASSERT_DOUBLE_EQ(actual, column[i]);
  }
}

TEST(JitCompiler, table_filter) {
  auto schema = simd::TableSchema::GetOrCreate("test_filter_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
  });

  Context ctx;
  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
  }

  auto table = schema->NewTable(ctx);

  std::ignore = table->Set("id", std::move(ids));
  std::ignore = table->Set("city", std::move(city));

  std::string expr = R"(
    table.filter(table.city=="sz")
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<simd::Table*, Context&, simd::Table*>(
      expr, {{"_"}, {"table", "test_filter_table"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  simd::Table* new_table = f(ctx, table.get());
  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();

  int expect_id = 10;
  for (size_t i = 0; i < ids.size(); i++) {
    ASSERT_EQ(new_id_column[i], expect_id);
    ASSERT_EQ(new_city_column[i], "sz");
    expect_id += 4;
  }
}
struct TestUser {
  int id;
  double score;
};

TEST(JitCompiler, table_order_by) {
  auto schema = simd::TableSchema::GetOrCreate("table_order_by_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
    std::ignore = s->AddColumn<double>("score");
    std::ignore = s->AddColumn<Pointer>("user");
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  std::vector<double> scores;
  std::vector<TestUser> users;

  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
    scores.emplace_back(1.1 + i);
    TestUser user;
    user.id = ids[ids.size() - 1];
    user.score = scores[scores.size() - 1];
    users.emplace_back(user);
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("id", std::move(ids));
  std::ignore = table->Set("city", std::move(city));
  std::ignore = table->Set("score", std::move(scores));
  std::ignore = table->Set("user", users);
  std::string expr = R"(
    table.order_by(table.score, true)
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<simd::Table*, simd::Table*>(expr, {{"table", "table_order_by_table"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  simd::Table* new_table = f(table.get());

  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();
  auto new_score_column = new_table->Get<double>("score").value();
  auto new_user_column = new_table->Get<Pointer>("user").value();
  ASSERT_EQ(N, new_city_column.Size());
  ASSERT_EQ(new_id_column.Size(), new_city_column.Size());

  for (size_t i = 0; i < N; i++) {
    RUDF_INFO("{} {} {}", new_id_column[i], new_city_column[i], new_score_column[i]);
    ASSERT_EQ(new_id_column[i], new_user_column[i].As<TestUser>()->id);
    ASSERT_DOUBLE_EQ(new_score_column[i], new_user_column[i].As<TestUser>()->score);
  }
}

TEST(JitCompiler, table_topk) {
  auto schema = simd::TableSchema::GetOrCreate("test_topk_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
    std::ignore = s->AddColumn<double>("score");
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  std::vector<double> scores;
  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
    scores.emplace_back(1.1 + i);
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("id", std::move(ids));
  std::ignore = table->Set("city", std::move(city));
  std::ignore = table->Set("score", std::move(scores));

  std::string expr = R"(
    table.topk(table.score,5,true)
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<simd::Table*, simd::Table*>(expr, {{"table", "test_topk_table"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  simd::Table* new_table = f(table.get());

  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();
  auto new_score_column = new_table->Get<double>("score").value();
  ASSERT_EQ(5, new_city_column.Size());
  ASSERT_EQ(new_id_column.Size(), new_city_column.Size());

  for (size_t i = 0; i < new_id_column.Size(); i++) {
    RUDF_INFO("{} {} {}", new_id_column[i], new_city_column[i], new_score_column[i]);
  }
}

TEST(JitCompiler, table_take) {
  auto schema = simd::TableSchema::GetOrCreate("test_take_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
    std::ignore = s->AddColumn<double>("score");
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  std::vector<double> scores;
  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
    scores.emplace_back(1.1 + i);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("id", ids);
  std::ignore = table->Set("city", city);
  std::ignore = table->Set("score", scores);

  std::string expr = R"(
      table<test_take_table> test_func(table<test_take_table> x){
      return x.head(5);
    }
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileFunction<simd::Table*, simd::Table*>(expr);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  simd::Table* new_table = f(table.get());
  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();
  auto new_score_column = new_table->Get<double>("score").value();

  ASSERT_EQ(5, new_id_column.Size());
  ASSERT_EQ(5, new_city_column.Size());
  ASSERT_EQ(5, new_score_column.Size());

  for (size_t i = 0; i < new_id_column.Size(); i++) {
    RUDF_INFO("{} {} {}", new_id_column[i], new_city_column[i], new_score_column[i]);
  }
}

TEST(JitCompiler, group_by) {
  auto schema = simd::TableSchema::GetOrCreate("test_group_by_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
    std::ignore = s->AddColumn<double>("score");
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  std::vector<double> scores;
  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
    scores.emplace_back(1.1 + i);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("id", ids);
  std::ignore = table->Set("city", city);
  std::ignore = table->Set("score", scores);

  auto table_group = table->GroupBy("city");
  ASSERT_EQ(table_group.size(), 4);
}

TEST(JitCompiler, dedup) {
  auto schema = simd::TableSchema::GetOrCreate("test_dedup_table", [&](simd::TableSchema* s) {
    std::ignore = s->AddColumn<int>("id");
    std::ignore = s->AddColumn<StringView>("city");
    std::ignore = s->AddColumn<double>("score");
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<std::string> city;
  std::vector<int> ids;
  std::vector<double> scores;
  for (size_t i = 0; i < N; i++) {
    ids.emplace_back(i + 10);
    city.emplace_back(candidate_citys[i % candidate_citys.size()]);
    scores.emplace_back(1.1 + i);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->Set("id", ids);
  std::ignore = table->Set("city", city);
  std::ignore = table->Set("score", scores);

  auto after_dedup = table->Filter(table->Dedup("city", 2));

  ctx.NewString("aaa{}", 1);
  ASSERT_EQ(after_dedup->Count(), 8);
}