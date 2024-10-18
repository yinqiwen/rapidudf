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

#include "rapidudf/meta/exception.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
int main() {
  spdlog::set_level(spdlog::level::debug);

  // 2. UDF string
  std::string source = R"(
    simd_vector<f32> boost_scores(Context ctx, simd_vector<string_view> location, simd_vector<f32> score) 
    { 
      auto boost=(location=="home"?2.0_f32:0_f32);
      return score*boost;
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  // 'rapidudf::Context' 是在simd 实现中必须携带的参数，涉及arena内存分配
  auto result =
      compiler.CompileFunction<simd::Vector<float>, rapidudf::Context&, simd::Vector<StringView>, simd::Vector<float>>(
          source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 测试数据， 需要将原始数据转成列式数据
  std::vector<float> scores;
  std::vector<std::string> locations;
  for (size_t i = 0; i < 4096; i++) {
    scores.emplace_back(1.1 + i);
    locations.emplace_back(i % 3 == 0 ? "home" : "other");
  }

  // 5. 执行function
  rapidudf::Context ctx;
  auto f = std::move(result.value());
  try {
    auto new_scores = f(ctx, ctx.NewSimdVector(locations), ctx.NewSimdVector(scores));
    for (size_t i = 0; i < new_scores.Size(); i++) {
      // RUDF_INFO("{}", new_scores[i]);
    }
  } catch (rapidudf::UDFRuntimeException& ex) {
    // handle exception
  }

  return 0;
};