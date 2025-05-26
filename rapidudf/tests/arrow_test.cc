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

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>
#include <gtest/gtest.h>
#include <vector>
#include "rapidudf/context/context.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"
#include "rapidudf/types/vector.h"

using namespace rapidudf;

struct Test2 {
  static constexpr bool destructor_disabled = true;
  int a;
  int b;
};

template <typename T>
struct Base {
  static int xtest(T* t) { return t->test(); }
};

struct Test1 : public Base<Test1> {
  int test() { return 1; }
};

TEST(Arrow, simple) {
  arrow::BooleanBuilder builder;

  builder.Append(true);
  builder.Append(false);
  builder.Append(true);
  std::shared_ptr<arrow::BooleanArray> out;
  builder.Finish(&out);

  fmt::print("###length:{}  child_data:{} buffers:{}\n", out->length(), out->data()->child_data.size(),
             out->data()->buffers.size());

  fmt::print("###buffer0:{}, mutable:{}\n", out->data()->buffers[1]->size(),
             out->data()->buffers[1]->mutable_data() != nullptr);

  // auto dtyep = rapidudf::get_arrow_dtype<std::vector<int>>();
  // RUDF_INFO("###dtype:{}\n", dtyep == nullptr);

  std::vector<int> a{1, 2, 3};

  auto result = Vector<int>::Wrap(default_arrow_memory_pool(), a, false);
  RUDF_INFO("###vector ok:{} {} \n", result.ok(), result.value().Size());
  const int* p = result.value().Data();
  RUDF_INFO("###{} {} {}\n", p[0], p[1], p[2]);

  std::vector<bool> b{false, true, false};
  auto result1 = Vector<bool>::Make(default_arrow_memory_pool(), b);
  RUDF_INFO("###{} {} {}\n", result1.value().Value(0), result1.value().Value(1), result1.value().Value(2));

  auto bb = result1.value();
  auto bbb = bb.Slice(1, 1);
  RUDF_INFO("###{} {} \n", bbb.Size(), bbb.Value(0));

  auto atest = Vector<int>::Make(default_arrow_memory_pool(), 100);
  RUDF_INFO("###{} \n", atest->Size());

  std::vector<int*> ps;
  auto ptest = Vector<int*>::Make(nullptr, ps);

  std::vector<std::string> ss{"hello", "world", "aaaaaa", "aaaaa", "aaaaa"};
  auto vec = Vector<std::string>::Make(nullptr, ss).value();
  StringView sss = vec.Value(1);
  auto zero = vec.Slice(0, 0);
  RUDF_INFO("###{} capcaity:{} {}/{}\n", sss, vec.BytesCapacity(), zero.Size(), zero.BytesCapacity());

  const StringView* xps = vec.Data();

  vec.Slice(0, 1);
}

TEST(Arrow, ptr) {
  std::vector<const uint8_t*> ptrs;

  auto x = Pointer::Wrap(ptrs);
  std::vector<Pointer> y;
  Vector<Pointer>::Wrap(nullptr, x);
  Vector<Pointer>::Wrap(nullptr, y);

  Context ctx;
  ctx.NewVector(x);
  ctx.NewVector(y);
}