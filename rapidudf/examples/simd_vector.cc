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

#include "rapidudf/types/simd_vector.h"
#include <cmath>
#include <string>
#include "rapidudf/context/context.h"
#include "rapidudf/rapidudf.h"

static float wilson_ctr(float click, float show, float z = 1.96) {
  const float epsilon = 1e-6;
  if (click > show) show = click;
  float p = click / (show + epsilon);
  if (show < epsilon) return 0.0;
  float n = show;
  float A = p + z * z / (2 * n);
  float B = std::sqrt(p * (1 - p) / n + z * z / (4 * (n * n)));
  float C = z * B;
  float D = 1 + z * z / n;
  float ctr = (A - C) / D;
  return ctr;
}

int main() {
  // 1. 如果需要, 可以设置rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);

  // 2. UDF string
  std::string source = R"(
    simd_vector<f32> wilson_ctr(Context ctx, simd_vector<f32> click, simd_vector<f32> show, f32 z) 
    { 
       var epsilon = 0.000001_f32;
       //if (click > show) show = click;
       show = (click>show)?click:show;
       var p = click/(show+epsilon);
       var n = show;
       var A = p + z * z / (2 * n);
       var B = sqrt(p * (1 - p) / n + z * z / (4 * (n * n)));
       var C = z * B;
       var D = 1 + z * z / n;
       var ctr = (A - C) / D;
       // if (show < epsilon) return 0.0;
       ctr = (show < epsilon)?0_f32:ctr;
       return ctr;
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  rapidudf::Context ctx;
  using simd_vector_f32 = rapidudf::simd::Vector<float>;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<simd_vector_f32, rapidudf::Context&, simd_vector_f32, simd_vector_f32, float>(
      source, true);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // // 4. 执行function
  // rapidudf::JitFunction<int, int> f = std::move(result.value());

  return 0;
};