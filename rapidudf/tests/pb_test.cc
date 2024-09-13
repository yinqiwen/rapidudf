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

#include "rapidudf/rapidudf.h"
#include "rapidudf/tests/test_pb.pb.h"

using namespace rapidudf;
using namespace rapidudf::ast;

// RUDF_STRUCT_MEMBER_METHODS(::test::Item, id)
RUDF_PB_FIELDS(::test::Header, id, scene, items, item_map, mapping, items_size)
RUDF_PB_FIELDS(::test::Item, id)

RUDF_PB_SET_FIELDS(::test::Header, id, scene)

TEST(JitCompiler, pb_access_read_int) {
  spdlog::set_level(spdlog::level::debug);

  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");

  JitCompiler compiler;
  std::string source = R"(
    int test_func(test::Header x){
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test::Header&>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(pb_header), pb_header.id());
  pb_header.set_id(2000);
  ASSERT_EQ(f(pb_header), pb_header.id());
}
TEST(JitCompiler, pb_access_read_str) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test::Header x){
      return x.scene();
    }
   )";
  auto rc = compiler.CompileFunction<StringView, const test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header), "hello,world");
}
TEST(JitCompiler, pb_access_write_int) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::Header x, int y){
      x.set_id(y);
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, test::Header*, int>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header, 1024), 1024);
  ASSERT_EQ(pb_header.id(), 1024);
}

struct A {
  template <typename T = int>
  A(T n = {}) {}
};

TEST(JitCompiler, pb_read_repetead) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");
  pb_header.add_items()->set_id(10001);

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::Header x){
      return x.items().get(0).id();
    }
   )";
  auto rc = compiler.CompileFunction<int, test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header), 10001);
}

TEST(JitCompiler, pb_write_string) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");
  pb_header.add_items()->set_id(10001);

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test::Header x){
      x.set_scene("123456");
      return x.scene();
    }
   )";
  auto rc = compiler.CompileFunction<StringView, test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header), "123456");
  ASSERT_EQ(pb_header.scene(), "123456");
}

TEST(JitCompiler, pb_read_map) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");
  pb_header.mutable_mapping()->insert({"k0", 100});
  pb_header.mutable_mapping()->insert({"k1", 101});

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::Header x){
      return x.mapping().get("k1");
    }
   )";
  auto rc = compiler.CompileFunction<int, test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header), 101);
}

TEST(JitCompiler, pb_read_map_item) {
  spdlog::set_level(spdlog::level::debug);
  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");
  ::test::Item item0;
  item0.set_id(1001);
  pb_header.mutable_item_map()->insert({"k0", item0});

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::Header x, string_view key){
      return x.item_map().get(key).id();
    }
   )";
  auto rc = compiler.CompileFunction<int, test::Header*, StringView>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header, "k0"), 1001);
  ASSERT_EQ(f(&pb_header, "k1"), 0);
}