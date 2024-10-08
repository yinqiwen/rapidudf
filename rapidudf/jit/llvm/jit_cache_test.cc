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
#include "rapidudf/jit/llvm/jit_cache.h"
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;
using namespace rapidudf::llvm;

TEST(JitCompiler, func_cache) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    bool test_func(json x){
      return x["key"] == 123;
    }
  )";
  auto rc = JitCompilerCache::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));

  rc = JitCompilerCache::GetFunction<bool, const JsonObject&>(content);
  ASSERT_TRUE(rc.ok());
  f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}

TEST(JitCompiler, expr_cache) {
  spdlog::set_level(spdlog::level::debug);
  std::vector<int> vec{1, 2, 3};
  JitCompiler compiler;
  JsonObject json;
  json["key"] = 123;

  std::string content = R"(
    x["key"] == 123
  )";
  auto rc = JitCompilerCache::GetExpression<bool, const JsonObject&>(content, {"x"});
  if (!rc.ok()) {
    RUDF_ERROR("{}", rc.status().ToString());
  }
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(json));

  rc = JitCompilerCache::GetExpression<bool, const JsonObject&>(content, {"x"});
  ASSERT_TRUE(rc.ok());
  f = std::move(rc.value());
  ASSERT_TRUE(f(json));
}