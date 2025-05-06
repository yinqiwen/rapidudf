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

struct FilterStruct {
  std::string city;
  int id;
  double score;
  std::array<float, 64> pad0;
  std::vector<float> pad1;
};
RUDF_STRUCT_FIELDS(FilterStruct, city, id, score, pad0, pad1)

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