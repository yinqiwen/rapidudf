// /*
//  * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include <gtest/gtest.h>
#include <array>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/executors/thread_pool.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;

struct SimpleStruct {
  std::string sv;
  float fv;
  int iv;
  bool bv;
};
RUDF_STRUCT_FIELDS(SimpleStruct, sv, fv, iv, bv)

TEST(JitCompiler, table_simple) {
  auto schema = table::TableSchema::GetOrCreate(
      "mytable", [](table::TableSchema* s) { std::ignore = s->AddColumns<SimpleStruct>(); });
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::vector<SimpleStruct> objs;
  objs.emplace_back(SimpleStruct{"a0", 1, 10, false});
  objs.emplace_back(SimpleStruct{"a1", 2, 20, true});
  objs.emplace_back(SimpleStruct{"a2", 3.1, 30, false});
  objs.emplace_back(SimpleStruct{"a3", 5.6, 40, true});

  auto status = table->AddRows(objs);
  ASSERT_TRUE(status.ok());

  auto fv_result = table->Get<float>("fv");
  ASSERT_TRUE(fv_result.ok());
  ASSERT_EQ(fv_result.value().Size(), objs.size());
  for (size_t i = 0; i < objs.size(); i++) {
    ASSERT_FLOAT_EQ(fv_result.value()[i], objs[i].fv);
  }

  auto sv_result = table->Get<StringView>("sv");
  ASSERT_TRUE(sv_result.ok());
  ASSERT_EQ(sv_result.value().Size(), objs.size());
  // ASSERT_TRUE(sv_result.value().IsReadonly());
  for (size_t i = 0; i < objs.size(); i++) {
    ASSERT_EQ(sv_result.value()[i], StringView(objs[i].sv));
  }

  auto bv_result = table->Get<bool>("bv");
  ASSERT_TRUE(bv_result.ok());
  ASSERT_EQ(bv_result.value().Size(), objs.size());
  for (size_t i = 0; i < objs.size(); i++) {
    ASSERT_EQ(bv_result.value()[i], Bit(objs[i].bv));
  }
}

struct ComplexStruct {
  double Click;
  double Like;
  double Join;
  double Inter;
  double TimeV1;
  double PostComment;
  double PositiveCommentV1;
  double ExpoTimeV1;
};
RUDF_STRUCT_FIELDS(ComplexStruct, Click, Like, Join, Inter, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1)

TEST(JitCompiler, table_func1) {
  auto schema = table::TableSchema::GetOrCreate(
      "score_table", [](table::TableSchema* s) { std::ignore = s->AddColumns<ComplexStruct>(); });

  std::vector<ComplexStruct> objs;
  objs.emplace_back(ComplexStruct{1, 1, 1, 1, 1, 1, 1, 1});
  objs.emplace_back(ComplexStruct{2, 2, 2, 2, 2, 2, 2, 2});
  objs.emplace_back(ComplexStruct{3, 3, 3, 3, 3, 3, 3, 3});
  objs.emplace_back(ComplexStruct{5, 5, 5, 5, 5, 5, 5, 5});

  Context ctx;
  auto table = schema->NewTable(ctx);
  auto status = table->AddRows(objs);
  ASSERT_TRUE(status.ok());

  std::string multiple_pow =
      "(Click^10.0)*((Like+0.000082)^4.7)*(Inter^3.5)*((Join+0.000024)^5.5)*(TimeV1^7.0)*((PostComment+0.000024)^3.5)*("
      "(PositiveCommentV1+0.0038)^1.0)*(ExpoTimeV1^1.5)";

  std::string content = R"(
(table.Click^10.0)*((table.Like+0.000082)^4.7)*(table.Inter^3.5)*((table.Join+0.000024)^5.5)*(table.TimeV1^7.0)*((table.PostComment+0.000024)^3.5)*((table.PositiveCommentV1+0.0038)^1.0)*(table.ExpoTimeV1^1.5)
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<Vector<double>, Context&, table::Table&>(
      content, {{"_"}, {"table", "score_table"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  auto column = f(ctx, *table);
  //   Vector<double> result = column->ToVector<double>().value();
  for (size_t i = 0; i < objs.size(); i++) {
    double actual = std::pow(objs[i].Click, 10.0) * std::pow(objs[i].Like + 0.000082, 4.7) *
                    std::pow(objs[i].Inter, 3.5) * std::pow(objs[i].Join + 0.000024, 5.5) *
                    std::pow(objs[i].TimeV1, 7.0) * std::pow(objs[i].PostComment + 0.000024, 3.5) *
                    std::pow(objs[i].PositiveCommentV1 + 0.0038, 1.0) * std::pow(objs[i].ExpoTimeV1, 1.5);
    ASSERT_DOUBLE_EQ(actual, column[i]);
  }
}

struct TestFilterStruct {
  int id;
  std::string city;
};
RUDF_STRUCT_FIELDS(TestFilterStruct, id, city)

TEST(JitCompiler, table_filter) {
  auto schema = table::TableSchema::GetOrCreate(
      "test_filter_table", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestFilterStruct>(); });

  Context ctx;
  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestFilterStruct> objs;

  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestFilterStruct{(static_cast<int>(i) + 10), candidate_citys[i % candidate_citys.size()]});
  }
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  std::string expr = R"(
    table.filter(table.city=="sz")
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<table::Table*, Context&, table::Table*>(
      expr, {{"_"}, {"table", "test_filter_table"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  table::Table* new_table = f(ctx, table.get());
  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();

  int expect_id = 10;
  for (size_t i = 0; i < new_id_column.Size(); i++) {
    ASSERT_EQ(new_id_column[i], expect_id);
    ASSERT_EQ(new_city_column[i], "sz");
    expect_id += 4;
  }
}
struct TestUser {
  int id;
  double score;
  std::string city;
  int repeate = 0;
};
RUDF_STRUCT_FIELDS(TestUser, id, score, city, repeate)
TEST(JitCompiler, table_order_by) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs;
  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  std::string expr = R"(
    table.order_by(table.score, true)
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<table::Table*, table::Table*>(expr, {{"table", "TestUser"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  table::Table* new_table = f(table.get());

  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();
  auto new_score_column = new_table->Get<double>("score").value();
  ASSERT_EQ(N, new_city_column.Size());
  ASSERT_EQ(new_id_column.Size(), new_city_column.Size());

  for (size_t i = 0; i < N; i++) {
    ASSERT_EQ(new_id_column[i], new_table->SlowGetRow<TestUser>(i)->id);
    ASSERT_DOUBLE_EQ(new_score_column[i], new_table->SlowGetRow<TestUser>(i)->score);
  }
}

TEST(JitCompiler, table_topk) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs;
  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  std::string expr = R"(
    table.topk(table.score,5,true)
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileDynObjExpression<table::Table*, table::Table*>(expr, {{"table", "TestUser"}});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  table::Table* new_table = f(table.get());

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
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs;
  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  std::string expr = R"(
      table<TestUser> test_func(table<TestUser> x){
      return x.head(5);
    }
  )";

  JitCompiler compiler;
  auto rc = compiler.CompileFunction<table::Table*, table::Table*>(expr);
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  table::Table* new_table = f(table.get());
  auto new_id_column = new_table->Get<int>("id").value();
  auto new_city_column = new_table->Get<StringView>("city").value();
  auto new_score_column = new_table->Get<double>("score").value();

  ASSERT_EQ(5, new_table->Count());
  ASSERT_EQ(5, new_id_column.Size());
  ASSERT_EQ(5, new_city_column.Size());
  ASSERT_EQ(5, new_score_column.Size());

  for (size_t i = 0; i < new_id_column.Size(); i++) {
    RUDF_INFO("{} {} {}", new_id_column[i], new_city_column[i], new_score_column[i]);
  }
}

TEST(JitCompiler, group_by) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs;
  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  auto table_group = table->GroupBy("city");
  ASSERT_EQ(table_group.size(), 4);
}

TEST(JitCompiler, dedup) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs;
  for (size_t i = 0; i < N; i++) {
    objs.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(objs);

  auto after_dedup = table->Filter(table->Dedup("city", 2));

  ASSERT_EQ(after_dedup->Count(), 8);
}

struct FilterStruct {
  std::string city;
  int id;
  double score;
};
RUDF_STRUCT_FIELDS(FilterStruct, city, id, score)

TEST(JitCompiler, filter) {
  auto schema = table::TableSchema::GetOrCreate(
      "test_filter_table1", [&](table::TableSchema* s) { std::ignore = s->AddColumns<FilterStruct>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<FilterStruct> items;
  for (size_t i = 0; i < N; i++) {
    FilterStruct item;
    item.city = candidate_citys[i % candidate_citys.size()];
    item.id = i + 10;
    item.score = 1.1 + i;
    items.emplace_back(item);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(items);

  auto filter_bits =
      table->Filter<FilterStruct>([](size_t i, const FilterStruct* item) -> bool { return item->city == "bj"; });
  ASSERT_EQ(filter_bits.CountTrue(), 25);
}
struct TransformStruct {
  std::string city;
  int id;
  double score;
};
RUDF_STRUCT_FIELDS(TransformStruct, city, id, score)
TEST(JitCompiler, map) {
  auto tranform_schema = table::TableSchema::GetOrCreate(
      "test_map_transform", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TransformStruct>(); });
  auto schema = table::TableSchema::GetOrCreate(
      "test_filter_table1", [&](table::TableSchema* s) { std::ignore = s->AddColumns<FilterStruct>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<FilterStruct> items;
  for (size_t i = 0; i < N; i++) {
    FilterStruct item;
    item.city = candidate_citys[i % candidate_citys.size()];
    item.id = i + 10;
    item.score = 1.1 + i;
    items.emplace_back(item);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(items);

  auto transform_table_result = table->Map<TransformStruct, FilterStruct>(
      tranform_schema, [&](size_t i, const FilterStruct* s) -> TransformStruct* {
        if (i == 50) {
          return nullptr;
        }
        auto p = table->GetContext().New<TransformStruct>();
        return p;
      });
  ASSERT_TRUE(transform_table_result.value());
  auto transform_table = std::move(transform_table_result.value());
  ASSERT_EQ(transform_table->Count(), N - 1);
}

TEST(JitCompiler, flat_map) {
  auto tranform_schema = table::TableSchema::GetOrCreate(
      "test_map_transform", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TransformStruct>(); });
  auto schema = table::TableSchema::GetOrCreate(
      "test_filter_table1", [&](table::TableSchema* s) { std::ignore = s->AddColumns<FilterStruct>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<FilterStruct> items;
  for (size_t i = 0; i < N; i++) {
    FilterStruct item;
    item.city = candidate_citys[i % candidate_citys.size()];
    item.id = i + 10;
    item.score = 1.1 + i;
    items.emplace_back(item);
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  std::ignore = table->AddRows(items);

  auto transform_table_result =
      table->FlatMap<TransformStruct, FilterStruct>(tranform_schema, [&](size_t i, const FilterStruct* s) {
        std::vector<TransformStruct*> vec;
        for (int i = 0; i < 3; i++) {
          vec.emplace_back(table->GetContext().New<TransformStruct>());
        }
        return vec;
      });
  ASSERT_TRUE(transform_table_result.value());
  auto transform_table = std::move(transform_table_result.value());
  ASSERT_EQ(transform_table->Count(), N * 3);
}

TEST(JitCompiler, distinct) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs1, objs2;
  for (size_t i = 0; i < N; i++) {
    objs1.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
    objs2.emplace_back(
        TestUser{static_cast<int>(i + 10 + N / 2), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table1 = schema->NewTable(ctx);
  auto table2 = schema->NewTable(ctx);
  std::ignore = table1->AddRows(objs1);
  std::ignore = table1->AddRows(objs2);
  auto table3 = table1->Concat(table2);
  ASSERT_EQ(table3->Count(), 2 * N);

  absl::Status s = table3->Distinct("id");
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(table3->Count(), 150);
}

TEST(JitCompiler, distinct_merge) {
  auto schema = table::TableSchema::GetOrCreate(
      "TestUser", [&](table::TableSchema* s) { std::ignore = s->AddColumns<TestUser>(); });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<TestUser> objs1, objs2;
  for (size_t i = 0; i < N; i++) {
    objs1.emplace_back(TestUser{static_cast<int>(i + 10), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
    objs2.emplace_back(
        TestUser{static_cast<int>(i + 10 + N / 2), 1.1 + i, candidate_citys[i % candidate_citys.size()]});
  }

  Context ctx;
  auto table1 = schema->NewTable(ctx);
  auto table2 = schema->NewTable(ctx);
  std::ignore = table1->AddRows(objs1);
  std::ignore = table2->AddRows(objs2);
  auto table3 = table1->Concat(table2);
  ASSERT_EQ(table3->Count(), 2 * N);

  absl::Status s = table3->Distinct<TestUser>(std::vector<StringView>{"id"},
                                              [](TestUser* current, const TestUser* duplicate) -> TestUser* {
                                                current->repeate++;
                                                return current;
                                              });
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(table3->Count(), 150);
  for (size_t i = 0; i < table3->Count(); i++) {
    RUDF_INFO("[{}] id:{},repeat:{}]", i, table3->SlowGetRow<TestUser>(i)->id,
              table3->SlowGetRow<TestUser>(i)->repeate);
  }
}

struct User1 {
  std::string name1;
  int id1;
};
RUDF_STRUCT_FIELDS(User1, name1, id1)
struct User2 {
  std::string name2;
  int id2;
};
RUDF_STRUCT_FIELDS(User2, name2, id2)
TEST(JitCompiler, multi_schema) {
  auto schema = table::TableSchema::GetOrCreate("test_multi_schmea_table", [&](table::TableSchema* s) {
    std::ignore = s->AddColumns<User1>();
    std::ignore = s->AddColumns<User2>();
  });

  size_t N = 100;
  std::vector<std::string> candidate_citys{"sz", "sh", "bj", "gz"};
  std::vector<User1> items1;
  std::vector<User2> items2;
  for (size_t i = 0; i < N; i++) {
    items1.emplace_back(User1{"", static_cast<int>(i)});
    items2.emplace_back(User2{"", static_cast<int>(i + 10)});
  }
  Context ctx;
  auto table = schema->NewTable(ctx);
  auto status = table->AddRows(items1, items2);
  ASSERT_TRUE(status.ok());

  std::atomic<int> visit_count{0};
  status = table->Foreach<void, User1, User2>([&](size_t, const User1* a, const User2* item) { visit_count++; },
                                              get_global_thread_pool(1));

  ASSERT_EQ(visit_count.load(), table->Count());

  RUDF_INFO("{}", schema->ToString());
}