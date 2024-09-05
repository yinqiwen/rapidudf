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
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/tests/test_fbs_generated.h"

using namespace rapidudf;
using namespace rapidudf::ast;

RUDF_STRUCT_MEMBER_METHODS(::test_fbs::Item, id)
RUDF_STRUCT_MEMBER_METHODS(::test_fbs::Header, items, id, scene, tags)

TEST(JitCompiler, fbs_access_read_int) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::HeaderT header;
  header.id = 101;
  header.scene = "hello,world";

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::Header::Pack(fbb, &header));
  const test_fbs::Header* fbs_ptr = test_fbs::GetHeader(fbb.GetBufferPointer());

  using VEC = flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>;
  rapidudf::try_register_fbs_vector_member_funcs<VEC>();

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test_fbs::Header x){
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test_fbs::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), header.id);
}

TEST(JitCompiler, fbs_read_vector_string) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::HeaderT header;
  header.id = 101;
  header.scene = "hello,world";
  header.tags.emplace_back("1111");

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::Header::Pack(fbb, &header));
  const test_fbs::Header* fbs_ptr = test_fbs::GetHeader(fbb.GetBufferPointer());

  using VEC = flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>;
  rapidudf::try_register_fbs_vector_member_funcs<VEC>();

  JitCompiler compiler;
  std::string content = R"(
    string_view test_func(test_fbs::Header x){
      return x.tags().get(0);
    }
   )";
  auto rc = compiler.CompileFunction<StringView, const test_fbs::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), "1111");
}

TEST(JitCompiler, fbs_read_vector_fbs) {
  spdlog::set_level(spdlog::level::debug);
  test_fbs::HeaderT header;
  header.id = 101;
  header.scene = "hello,world";
  flatbuffers::unique_ptr<test_fbs::ItemT> item(new test_fbs::ItemT);
  item->id = 1001;
  header.items.emplace_back(std::move(item));

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::Header::Pack(fbb, &header));
  const test_fbs::Header* fbs_ptr = test_fbs::GetHeader(fbb.GetBufferPointer());

  using VEC = flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>;
  rapidudf::try_register_fbs_vector_member_funcs<VEC>();

  JitCompiler compiler;
  std::string content = R"(
    u32 test_func(test_fbs::Header x){
      return x.items().get(0).id();
    }
   )";
  auto rc = compiler.CompileFunction<uint32_t, const test_fbs::Header*>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f(fbs_ptr), 1001);
}

TEST(JitCompiler, fbs_access_read_str) {
  spdlog::set_level(spdlog::level::debug);

  test_fbs::HeaderT header;
  header.id = 101;
  header.scene = "hello,world";

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(test_fbs::Header::Pack(fbb, &header));
  const test_fbs::Header* fbs_ptr = test_fbs::GetHeader(fbb.GetBufferPointer());

  JitCompiler compiler;
  std::string content = R"(
    bool test_func(test_fbs::Header x){
      return x.scene() == "hello,world";
    }
   )";
  auto rc = compiler.CompileFunction<bool, const test_fbs::Header*>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<std::string_view, const test_fbs::Header*>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(fbs_ptr));
}
// TEST(JitCompiler, pb_access_write_int) {
//   spdlog::set_level(spdlog::level::debug);
//   ::test::Header pb_header;
//   pb_header.set_scene("");
//   pb_header.set_id(101);
//   pb_header.set_scene("hello,world");

//   JitCompiler compiler;
//   std::string content = R"(
//     int test_func(test::Header x, int y){
//       x.set_id(y);
//       return x.id();
//     }
//    )";
//   auto rc = compiler.CompileFunction(content);
//   ASSERT_TRUE(rc.ok());
//   auto f = compiler.GetFunc<int, test::Header*, int>(true);
//   ASSERT_TRUE(f != nullptr);
//   ASSERT_EQ(f(&pb_header, 1024), 1024);
//   ASSERT_EQ(pb_header.id(), 1024);
// }