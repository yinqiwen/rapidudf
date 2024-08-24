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
#include "rapidudf/reflect/struct_access.h"
#include "rapidudf/tests/test_pb.pb.h"

using namespace rapidudf;
using namespace rapidudf::ast;

RUDF_STRUCT_MEMBER_METHODS(::test::Header, id, scene, set_id)

using TTT = void (test::Header::*)(const std::string&);

TEST(JitCompiler, pb_access_read_int) {
  spdlog::set_level(spdlog::level::debug);

  // TTT xyz = &::test::Header::set_scene;
  // RUDF_STRUCT_SAFE_MEMBER_METHOD_BIND(::test::Header, set_scene, xyz);

  ::test::Header pb_header;
  pb_header.set_scene("");
  pb_header.set_id(101);
  pb_header.set_scene("hello,world");

  JitCompiler compiler;
  std::string content = R"(
    int test_func(test::Header x){
      return x.id();
    }
   )";
  auto rc = compiler.CompileFunction<int, const test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<int, const test::Header*>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header), pb_header.id());
  pb_header.set_id(2000);
  ASSERT_EQ(f(&pb_header), pb_header.id());
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
  auto rc = compiler.CompileFunction<std::string_view, const test::Header*>(content);
  ASSERT_TRUE(rc.ok());
  // auto f = compiler.GetFunc<std::string_view, const test::Header*>(true);
  // ASSERT_TRUE(f != nullptr);
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
  // auto f = compiler.GetFunc<int, test::Header*, int>(true);
  // ASSERT_TRUE(f != nullptr);
  auto f = std::move(rc.value());
  ASSERT_EQ(f(&pb_header, 1024), 1024);
  ASSERT_EQ(pb_header.id(), 1024);
}