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
#include "rapidudf/examples/book_generated.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  // 1. 创建table schema
  auto schema = simd::TableSchema::GetOrCreate(
      "Book", [](simd::TableSchema* s) { std::ignore = s->BuildFromFlatbuffers<examples::Book>(); });

  // 2. UDF string
  std::string source = R"(
    table<Book> select_books(Context ctx, table<Book> x) 
    { 
       auto filtered = x.filter(x.price >90 && x.page_count<300 );
       // 降序排列
       return filtered.topk(filtered.price,10, true); 
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileFunction的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<simd::Table*, Context&, simd::Table*>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }
  auto f = std::move(result.value());

  // 4.1 测试数据
  std::vector<const examples::Book*> books;
  std::vector<std::unique_ptr<flatbuffers::FlatBufferBuilder>> fbs_buffers;
  for (size_t i = 0; i < 150; i++) {
    examples::BookT book;
    auto builder = std::make_unique<flatbuffers::FlatBufferBuilder>();
    book.price = (i + 10) % 150 + 1.1;
    book.title = "test_" + std::to_string(i);
    book.page_count = 80 + (i) % 200;
    book.category = i % 10 + 1;
    builder->Finish(examples::Book::Pack(*builder, &book));
    const examples::Book* fbs_ptr = examples::GetBook(builder->GetBufferPointer());
    books.emplace_back(fbs_ptr);
    fbs_buffers.emplace_back(std::move(builder));
  }
  // // 4.2 创建table实例
  rapidudf::Context ctx;
  auto table = schema->NewTable(ctx);
  // 4.3 填充数据
  std::ignore = table->BuildFromFlatbuffersVector(books);

  try {
    // 5. 执行function
    auto result_table = f(ctx, table.get());
    // 5.1 获取列
    auto result_price = result_table->Get<float>("price").value();
    auto result_title = result_table->Get<StringView>("title").value();
    auto result_category = result_table->Get<uint32_t>("category").value();
    auto result_page_count = result_table->Get<uint32_t>("page_count").value();

    for (size_t i = 0; i < result_price.Size(); i++) {
      RUDF_INFO("title:{},price:{},category:{},page_count:{}", result_title[i], result_price[i], result_category[i],
                result_page_count[i]);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }
  return 0;
};