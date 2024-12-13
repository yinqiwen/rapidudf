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

#include <vector>
#include "rapidudf/examples/student.pb.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 1. 创建table schema
  auto schema = table::TableSchema::GetOrCreate(
      "Student", [](table::TableSchema* s) { std::ignore = s->AddColumns<examples::Student>(); });

  // 2. UDF string
  std::string source = R"(
    table<Student> select_students(Context ctx, table<Student> x) 
    { 
       auto filtered = x.filter(x.score >90 && x.age<10);
       // 降序排列
      return filtered.topk(filtered.score,10, true); 
      // return filtered;
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<table::Table*, Context&, table::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 测试数据
  std::vector<examples::Student> students;
  for (size_t i = 0; i < 150; i++) {
    examples::Student student;
    student.set_score((i + 1) % 150);
    student.set_name("test_" + std::to_string(i));
    student.set_age(i % 5 + 8);
    students.emplace_back(std::move(student));
  }
  // 4.2 创建table实例
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  // 4.3 填充数据

  std::ignore = table->AddRows(students);

  try {
    // 5. 执行function
    auto result_table = f(ctx, table.get());
    // 5.1 获取列
    auto result_scores = result_table->Get<float>("score").value();
    auto result_names = result_table->Get<StringView>("name").value();
    auto result_ages = result_table->Get<int32_t>("age").value();

    for (size_t i = 0; i < result_scores.Size(); i++) {
      RUDF_INFO("name:{},score:{},age:{}", result_names[i], result_scores[i], result_ages[i]);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }
  return 0;
};