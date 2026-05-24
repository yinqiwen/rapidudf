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

#include <gtest/gtest.h>
#include <vector>

#include "rapidudf/functions/simd/string.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;
using namespace rapidudf::ast;
TEST(JitCompiler, string_size) {
  JitCompiler compiler;
  std::string content = R"(str.size())";
  auto rc = compiler.CompileExpression<size_t, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), 5);
  ASSERT_EQ(f("he"), 2);
}
TEST(JitCompiler, string_contains) {
  JitCompiler compiler;
  std::string content = R"(str.contains("hello"))";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), true);
  ASSERT_EQ(f("he"), false);
}
TEST(JitCompiler, string_starts_with) {
  JitCompiler compiler;
  std::string content = R"(str.starts_with("hello"))";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_EQ(f("hello"), true);
  ASSERT_EQ(f("he"), false);
}

// Regression: long (>12 char) string literals must be emitted into the JIT
// module itself (LLVM global constant) so the compiled function does not
// dangle on host-side std::string memory. We exercise both equality and
// contains over a 32-char literal that would overflow StringView's inline
// storage.
TEST(JitCompiler, long_string_literal_eq) {
  JitCompiler compiler;
  std::string content = R"(str == "abcdefghijklmnopqrstuvwxyz0123")";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(StringView("abcdefghijklmnopqrstuvwxyz0123")));
  ASSERT_FALSE(f(StringView("abcdefghijklmnopqrstuvwxyz0124")));
  ASSERT_FALSE(f(StringView("short")));
}

TEST(JitCompiler, long_string_literal_contains) {
  JitCompiler compiler;
  std::string content = R"(str.contains("abcdefghijklmnopqrstuvwxyz0123"))";
  auto rc = compiler.CompileExpression<bool, StringView>(content, {{"str"}});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  ASSERT_TRUE(f(StringView("xx_abcdefghijklmnopqrstuvwxyz0123_yy")));
  ASSERT_FALSE(f(StringView("abcdefghijklmnopqrstuvwxyz0124")));
}

TEST(JitCompiler, split_by_str) {
  std::string str = "abc;;cde;;eed;;";
  auto ss = functions::simd_string_split_by_string(str, ";;");

  ASSERT_EQ(ss.size(), 3);
  ASSERT_EQ(ss[0], "abc");
  ASSERT_EQ(ss[1], "cde");
  ASSERT_EQ(ss[2], "eed");
}

TEST(JitCompiler, split_by_char) {
  std::string str = "abc,cde,eed,,";
  auto ss = functions::simd_string_split_by_char(str, ',');

  ASSERT_EQ(ss.size(), 3);
  ASSERT_EQ(ss[0], "abc");
  ASSERT_EQ(ss[1], "cde");
  ASSERT_EQ(ss[2], "eed");
}

// Regression: split must include the last segment after the final separator.
TEST(StringSplit, LastSegmentPreserved) {
  // "a,b" split by ',' should yield ["a", "b"], not just ["a"]
  auto ss = functions::simd_string_split_by_char("a,b", ',');
  ASSERT_EQ(ss.size(), 2);
  ASSERT_EQ(ss[0], "a");
  ASSERT_EQ(ss[1], "b");

  // Multi-char separator: last segment preserved
  auto ss2 = functions::simd_string_split_by_string("a;;b;;c", ";;");
  ASSERT_EQ(ss2.size(), 3);
  ASSERT_EQ(ss2[0], "a");
  ASSERT_EQ(ss2[1], "b");
  ASSERT_EQ(ss2[2], "c");

  // No trailing separator: entire string is one segment
  auto ss3 = functions::simd_string_split_by_char("hello", ',');
  ASSERT_EQ(ss3.size(), 1);
  ASSERT_EQ(ss3[0], "hello");

  // Trailing separator with non-empty last segment
  auto ss4 = functions::simd_string_split_by_string("x;;y;;z", ";;");
  ASSERT_EQ(ss4.size(), 3);
  ASSERT_EQ(ss4[2], "z");
}

TEST(JitCompiler, find_by_str) {
  std::string str = "abc,cde,eed,,";
  auto pos = functions::simd_string_find_string(str, "cde");

  ASSERT_EQ(pos, 4);

  str = "eed;;";
  pos = functions::simd_string_find_string(str, ";;");
  ASSERT_EQ(pos, 3);
}

TEST(JitCompiler, find_by_ch) {
  std::string str = "abc,cde,eed,,";
  auto pos = functions::simd_string_find_char(str, ',');

  ASSERT_EQ(pos, 3);
}

// Regression: string find must not match zero-padded bytes in the tail.
TEST(StringFind, ShortStringNoFalseMatch) {
  // A very short string where the tail section is used.
  // "ab" does not contain 'c', should return -1.
  ASSERT_EQ(functions::simd_string_find_char("ab", 'c'), -1);
  ASSERT_EQ(functions::simd_string_find_string("ab", "cd"), -1);

  // Verify correct find on short strings
  ASSERT_EQ(functions::simd_string_find_char("abc", 'b'), 1);
  ASSERT_EQ(functions::simd_string_find_string("abc", "bc"), 1);
}

// Regression: find_string must find exact match when len == part_len.
TEST(StringFind, ExactMatch) {
  // "ab" find "ab" should return 0, not -1
  ASSERT_EQ(functions::simd_string_find_string("ab", "ab"), 0);
  ASSERT_EQ(functions::simd_string_find_string("hello", "hello"), 0);

  // Pattern at end of string
  ASSERT_EQ(functions::simd_string_find_string("xyzabc", "abc"), 3);
}

// Regression: count_true must not include padding bits.
TEST(BitsOps, CountTrueNoPadding) {
  // Create a vector of 10 bits: [1,1,1,1,1,1,1,1,1,0]
  // Byte 0 = 0xFF (bits 0-7 all set), Byte 1 = 0x01 (bit 8 set, bit 9 clear)
  // padding bits 10-15 must be ignored
  JitCompiler compiler;
  std::string content = R"(
    simd_vector<bit> test_func(Context ctx, simd_vector<i32> x){
      return x > 5;
    }
  )";
  auto rc = compiler.CompileFunction<Vector<Bit>, Context&, Vector<int>>(content);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  Context ctx;
  // Values: 1,2,3,4,5,6,7,8,9,10 -> bits: 0,0,0,0,0,1,1,1,1,1 -> 5 true
  std::vector<int> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto result = f(ctx, vals);
  ASSERT_EQ(result.Size(), 10);
  // Count true bits manually
  int count = 0;
  for (size_t i = 0; i < result.Size(); i++) {
    if (result[i]) count++;
  }
  EXPECT_EQ(count, 5);
}