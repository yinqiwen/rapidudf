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

#include "rapidudf/types/dyn_object.h"
#include <gtest/gtest.h>
#include <functional>
#include <string_view>
#include <tuple>
#include <vector>

#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;
TEST(JitCompiler, dyn_obj) {
  auto schema = DynObjectSchema::GetOrCreate("DynObj", [](DynObjectSchema* s) {
    std::ignore = s->AddField<StringView>("sv");
    std::ignore = s->AddField<float>("fv");
    std::ignore = s->AddField<int>("iv");
  });

  auto obj = schema->NewObject();
  std::ignore = obj->Set("fv", static_cast<float>(1.23));
  std::ignore = obj->Set("iv", static_cast<int>(1235));
  StringView constant_sv = "hello,world";
  std::ignore = obj->Set("sv", std::move(constant_sv));

  auto fv = obj->Get<float>("fv");
  ASSERT_TRUE(fv.ok());
  ASSERT_FLOAT_EQ(fv.value(), 1.23);

  auto iv = obj->Get<int>("iv");
  ASSERT_TRUE(iv.ok());
  ASSERT_EQ(iv.value(), 1235);

  auto sv = obj->Get<StringView>("sv");
  ASSERT_TRUE(sv.ok());
  ASSERT_EQ(sv.value(), constant_sv);
}

TEST(JitCompiler, dyn_obj_expression_access) {
  const DynObjectSchema* schema = DynObjectSchema::GetOrCreate("myobj", [](DynObjectSchema* s) {
    std::ignore = s->AddField<StringView>("sv");
    std::ignore = s->AddField<int>("iv");
  });
  auto obj = schema->NewObject();
  std::ignore = obj->Set("sv", "hello,world");
  std::ignore = obj->Set("iv", 1234);

  JitCompiler compiler;
  std::string content = R"(obj.sv)";
  auto rc0 = compiler.CompileDynObjExpression<StringView, DynObject&>(content, {{.name = "obj", .schema = "myobj"}});
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(*obj), "hello,world");

  auto rc1 = compiler.CompileDynObjExpression<int, DynObject&>("obj.iv", {{.name = "obj", .schema = "myobj"}});
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(*obj), 1234);
}

TEST(JitCompiler, dyn_obj_udf_access) {
  const DynObjectSchema* schema = DynObjectSchema::GetOrCreate("myobj", [](DynObjectSchema* s) {
    std::ignore = s->AddField<StringView>("sv");
    std::ignore = s->AddField<int>("iv");
  });
  auto obj = schema->NewObject();
  std::ignore = obj->Set("sv", "hello,world");
  std::ignore = obj->Set("iv", 1234);

  JitCompiler compiler;
  std::string content = R"(
   string_view test_func(dyn_obj<myobj> x){
      return x.sv;
    }
  )";
  auto rc0 = compiler.CompileFunction<StringView, DynObject&>(content);
  ASSERT_TRUE(rc0.ok());
  auto f0 = std::move(rc0.value());
  ASSERT_EQ(f0(*obj), "hello,world");

  content = R"(
   int test_func(dyn_obj<myobj> x){
      return x.iv;
    }
  )";
  auto rc1 = compiler.CompileFunction<int, DynObject&>(content);
  ASSERT_TRUE(rc1.ok());
  auto f1 = std::move(rc1.value());
  ASSERT_EQ(f1(*obj), 1234);
}