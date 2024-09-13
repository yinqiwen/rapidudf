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

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include "absl/strings/str_join.h"

#include "rapidudf/rapidudf.h"
#include "x86simdsort.h"

using namespace rapidudf;
using namespace rapidudf::ast;

TEST(JitCompiler, simd_sort) {
  std::vector<float> data{1.1, 2.2, 0.1, 0.3, 0.2, 11, 12, 3, 423, 12, 12312, 12, 4124};
  auto result = x86simdsort::argsort(data.data(), data.size(), false, false);
  RUDF_INFO("result:[{}], data:[{}]", absl::StrJoin(result, ","), absl::StrJoin(data, ","));

  auto result1 = x86simdsort::argselect(data.data(), 2, data.size(), false);
  RUDF_INFO("result:[{}], data:[{}]", absl::StrJoin(result1, ","), absl::StrJoin(data, ","));

  size_t n = 16;
  std::vector<int> vals;
  std::vector<float> keys;
  for (size_t i = 0; i < n; i++) {
    keys.emplace_back(100.0 - 1.1 - i);
    vals.emplace_back(i);
  }
  x86simdsort::keyvalue_qsort(keys.data(), vals.data(), keys.size());
  RUDF_INFO("key:[{}], val:[{}]", absl::StrJoin(keys, ","), absl::StrJoin(vals, ","));
}
