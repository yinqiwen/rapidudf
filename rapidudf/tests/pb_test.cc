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
#include "rapidudf/tests/test_pb.pb.h"

using namespace rapidudf;
using namespace rapidudf::ast;

// RUDF_STRUCT_MEMBER_METHODS(::test::Item, id)
RUDF_PB_FIELDS(::test::Item, id)
RUDF_PB_FIELDS(::test::PBStruct, id, str, ids_array, strs_array, item_array, item_map, item_map, str_int_map)

RUDF_PB_SET_FIELDS(::test::PBStruct, id, str)

TEST(JitCompiler, pb_access_read_int) {
  spdlog::set_level(spdlog::level::debug);

  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");

  JitCompiler compiler;
  std::string source = "x.id()";
  auto rc = compiler.CompileExpression<int, const test::PBStruct&>(source, {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb), pb.id());
  pb.set_id(2000);
  ASSERT_EQ(f(pb), pb.id());
}
TEST(JitCompiler, pb_access_read_str) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test::PBStruct x){
      return x.str();
    }
   )";
  auto rc = compiler.CompileFunction<StringView, const test::PBStruct*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb), "hello,world");
}
TEST(JitCompiler, pb_access_write_int) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::PBStruct x, int y){
      x.set_id(y);
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, test::PBStruct&, int>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb, 1024), 1024);
  ASSERT_EQ(pb.id(), 1024);
}

TEST(JitCompiler, pb_read_repetead) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");
  pb.add_item_array()->set_id(10001);

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::PBStruct x){
      return x.item_array().get(0).id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test::PBStruct&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb), 10001);
}

TEST(JitCompiler, pb_write_string) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;

  pb.set_id(101);
  pb.set_str("hello,world");
  pb.add_item_array()->set_id(10001);

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test::PBStruct x){
      x.set_str("123456");
      return x.str();
    }
   )";
  auto rc = compiler.CompileFunction<StringView, test::PBStruct&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb), "123456");
  ASSERT_EQ(pb.str(), "123456");
}

TEST(JitCompiler, pb_read_map) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");
  pb.mutable_str_int_map()->insert({"k0", 100});
  pb.mutable_str_int_map()->insert({"k1", 101});

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::PBStruct x){
      return x.str_int_map().get("k1");
    }
   )";
  auto rc = compiler.CompileFunction<int, const test::PBStruct&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb), 101);
}

TEST(JitCompiler, pb_read_map_item) {
  spdlog::set_level(spdlog::level::debug);
  ::test::PBStruct pb;
  pb.set_id(101);
  pb.set_str("hello,world");
  ::test::Item item0;
  item0.set_id(1001);
  pb.mutable_item_map()->insert({"k0", item0});

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::PBStruct x, string_view key){
      return x.item_map().get(key).id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test::PBStruct&, StringView>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb, "k0"), 1001);
  ASSERT_ANY_THROW(f(pb, "k1"));
}