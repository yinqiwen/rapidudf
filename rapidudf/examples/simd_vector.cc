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

struct User {
  std::string city;
};
RUDF_STRUCT_FIELDS(User, city)  // 绑定User类，可在UDF里访问city字段

struct Feed {
  std::string city;
  float score;
};
struct Feeds {
  rapidudf::simd::Vector<rapidudf::StringView> city;
  rapidudf::simd::Vector<float> score;
};
RUDF_STRUCT_FIELDS(Feeds, city, score)  // 绑定Feeds类，可在UDF里访问city/score字段

int main() {
  spdlog::set_level(spdlog::level::debug);
  // 2. UDF string
  std::string source = R"(
    void boost_scores(User user,Feeds feeds) 
    { 
      // 注意boost是个float数组
      var boost=(feeds.city==user.city?1_f32:0);
      feeds.score*=boost;
    } 
  )";

  // 3. 编译生成Function,这里生成的Function对象可以保存以供后续重复执行
  rapidudf::JitCompiler compiler;
  // CompileExpression的模板参数支持多个，第一个模板参数为返回值类型，其余为function参数类型
  auto result = compiler.CompileFunction<void, const User&, Feeds&>(source);
  if (!result.ok()) {
    RUDF_ERROR("{}", result.status().ToString());
    return -1;
  }

  // 4.1 测试数据， 需要将原始数据转成列式数据
  User user;
  user.city = "sz";
  std::vector<Feed> feeds;
  for (size_t i = 0; i < 1024; i++) {
    Feed feed;
    feed.city = (i % 2 == 0 ? "sz" : "bj");
    feed.score = i + 1.1;
    feeds.emplace_back(feed);
  }

  // 4.2 将原始数据转成列式数据
  std::vector<rapidudf::StringView> citys;
  std::vector<float> scores;
  for (auto& feed : feeds) {
    citys.emplace_back(feed.city);
    scores.emplace_back(feed.score);
  }
  Feeds column_feeds;
  column_feeds.city = citys;
  column_feeds.score = scores;

  // 5. 执行function
  rapidudf::JitFunction<void, const User&, Feeds&> f = std::move(result.value());
  f(user, column_feeds);

  return 0;
};