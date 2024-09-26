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
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "rapidudf/rapidudf.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/string_view.h"

using namespace rapidudf;
using namespace rapidudf::ast;

// TEST(JitCompiler, many_args0) {
//   JitCompiler compiler;
//   std::string content = R"(
//     int test_func(int a, string_view b, string_view c, string_view d){
//        return b.size() + c.size() + d.size();
//     }
//   )";
//   auto rc = compiler.CompileFunction<int, int, StringView, StringView, StringView>(content, true);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   StringView str = "hello,world";
//   ASSERT_EQ(f(1, str, str, str), str.size() * 3);
// }

// TEST(JitCompiler, many_args1) {
//   JitCompiler compiler;
//   std::string content = R"(
//     int test_func(int a, simd_vector<i32> b,  simd_vector<i32> c,  simd_vector<i32> d,  simd_vector<i32> e){
//        return b.size() + c.size() + d.size() + e.size();
//     }
//   )";
//   auto rc =
//       compiler.CompileFunction<int, int, simd::Vector<int>, simd::Vector<int>, simd::Vector<int>, simd::Vector<int>>(
//           content, true);
//   ASSERT_TRUE(rc.ok());
//   auto f = std::move(rc.value());
//   std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8};
//   ASSERT_EQ(f(1, x, x, x, x), x.size() * 4);
// }

TEST(JitCompiler, many_args2) {
  JitCompiler compiler;
  std::string content = R"(
    int test_func(int a, std_string_view b, std_string_view c, std_string_view d){
       return b.size() + c.size() + d.size();
    }
  )";
  auto rc = compiler.CompileFunction<int, int, std::string_view, std::string_view, std::string_view>(content, true);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  std::string_view str = "hello,world1";
  ASSERT_EQ(f(1, str, str, str), str.size() * 3);
}
