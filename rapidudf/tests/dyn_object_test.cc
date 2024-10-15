/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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