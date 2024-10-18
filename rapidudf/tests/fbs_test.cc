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
#include "flatbuffers/minireflect.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/tests/test_fbs_generated.h"

using namespace rapidudf;
using namespace rapidudf::ast;

RUDF_STRUCT_MEMBER_METHODS(::test_fbs::Item, id)
RUDF_STRUCT_MEMBER_METHODS(::test_fbs::FBSStruct, id, str, item, strs, items, ints)

TEST(JitCompiler, fbs_access_read_int) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::FBSStructT header;
  header.id = 101;
  header.str = "hello,world";
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::FBSStruct::Pack(fbb, &header));
  const test_fbs::FBSStruct* fbs_ptr = test_fbs::GetFBSStruct(fbb.GetBufferPointer());
  using VEC = flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>;
  rapidudf::try_register_fbs_vector_member_funcs<VEC>();

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test_fbs::FBSStruct x){
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test_fbs::FBSStruct*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), header.id);

  auto* tt = test_fbs::FBSStruct::MiniReflectTypeTable();

  auto s = flatbuffers::FlatBufferToString(fbb.GetBufferPointer(), tt);
  RUDF_INFO("####offset null:{} {}", tt->values == nullptr, s);
}

TEST(JitCompiler, fbs_read_vector_string) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::FBSStructT fbs;
  fbs.id = 101;
  fbs.str = "hello,world";
  fbs.strs.emplace_back("1111");

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::FBSStruct::Pack(fbb, &fbs));
  const test_fbs::FBSStruct* fbs_ptr = test_fbs::GetFBSStruct(fbb.GetBufferPointer());

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test_fbs::FBSStruct x){
      return x.strs().get(0);
    }
   )";
  auto rc = compiler.CompileFunction<StringView, const test_fbs::FBSStruct*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), "1111");
}

TEST(JitCompiler, fbs_read_vector_fbs) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::FBSStructT fbs;
  fbs.id = 101;
  fbs.str = "hello,world";
  flatbuffers::unique_ptr<test_fbs::ItemT> item(new test_fbs::ItemT);
  item->id = 1001;
  fbs.items.emplace_back(std::move(item));

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::FBSStruct::Pack(fbb, &fbs));
  const test_fbs::FBSStruct* fbs_ptr = test_fbs::GetFBSStruct(fbb.GetBufferPointer());

  JitCompiler compiler;
  std::string content = R"(
    u32 test_func(test_fbs::FBSStruct x){
      return x.items().get(0).id();
    }
   )";
  auto rc = compiler.CompileFunction<uint32_t, const test_fbs::FBSStruct*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), 1001);
}

TEST(JitCompiler, fbs_access_read_str) {
  spdlog::set_level(spdlog::level::debug);

  test_fbs::FBSStructT fbs;
  fbs.id = 101;
  fbs.str = "hello,world";

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::FBSStruct::Pack(fbb, &fbs));
  const test_fbs::FBSStruct* fbs_ptr = test_fbs::GetFBSStruct(fbb.GetBufferPointer());

  JitCompiler compiler;
  std::string content = R"(
    bool test_func(test_fbs::FBSStruct x){
      return x.str() == "hello,world";
    }
   )";
  auto rc = compiler.CompileFunction<bool, const test_fbs::FBSStruct*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(fbs_ptr));
}