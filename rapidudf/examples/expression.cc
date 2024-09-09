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
#include "rapidudf/rapidudf.h"

int main() {
  // 1. 如果需要, 可以设置rapidudf logger
  //   std::shared_ptr<spdlog::logger> mylogger;
  //   rapidudf::set_default_logger(mylogger);

  // 2. expression string
  std::string expression = "x >= 1 && y < 10";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileExpression<bool, int, int>(expression, {"x", "y"});
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4. 执行function
  rapidudf::JitFunction<bool, int, int> f = std::move(result.value());
  bool v = f(2, 3);  // true
  v = f(0, 1);       // false
  return 0;
};