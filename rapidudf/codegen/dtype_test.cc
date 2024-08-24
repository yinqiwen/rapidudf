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

#include "rapidudf/codegen/dtype.h"
#include <cxxabi.h>
#include <gtest/gtest.h>
#include <vector>
#include "rapidudf/log/log.h"

struct TestStruct {
  int a;
};

using namespace rapidudf;
// TEST(DType, simple) {
//   uint32_t dtype0 = get_dtype<int>();
//   ASSERT_EQ(dtype0, get_dtype<int>());
//   uint32_t dtype1 = get_dtype<int*>();
//   uint32_t dtype2 = get_dtype<decltype(((TestStruct*)0)->a)>();
//   ASSERT_EQ(dtype0, dtype2);
//   ASSERT_EQ(dtype0, dtype1 - DATA_PTR_BEGIN);

//   uint32_t dtype_str = get_dtype<std::string>();
//   ASSERT_EQ(dtype_str, get_dtype<std::string>());
//   uint32_t dtype_str_ptr = get_dtype<std::string*>();
//   ASSERT_EQ(dtype_str, dtype_str_ptr - DATA_PTR_BEGIN);

//   uint32_t dtype_t = get_dtype<TestStruct>();
//   ASSERT_EQ(dtype_t, get_dtype<TestStruct>());
//   // uint32_t dtype_t_ptr = get_dtype<TestStruct*>();
//   uint32_t dtype_t_ptr = get_dtype<decltype((TestStruct*)0)>();
//   ASSERT_EQ(dtype_t, dtype_t_ptr - DATA_PTR_BEGIN);
// }

TEST(DType, simple1) {
  auto void_dtype = get_dtype<void>();
  ASSERT_EQ(void_dtype, DATA_VOID);

  auto dtype0 = get_dtype<int>();
  ASSERT_EQ(dtype0, get_dtype<int>());
  auto dtype1 = get_dtype<int*>();
  auto dtype2 = get_dtype<decltype(((TestStruct*)0)->a)>();
  ASSERT_EQ(dtype0, dtype2);
  ASSERT_TRUE(dtype1.IsPtr());
  ASSERT_TRUE(dtype1.IsSameFundamentalType(dtype0));

  auto dtype_str = get_dtype<std::string>();
  ASSERT_EQ(dtype_str, get_dtype<std::string>());
  auto dtype_str_ptr = get_dtype<std::string*>();
  ASSERT_TRUE(dtype_str_ptr.IsPtr());
  ASSERT_TRUE(dtype_str_ptr.IsSameFundamentalType(dtype_str));

  auto dtype_t = get_dtype<TestStruct>();
  ASSERT_EQ(dtype_t, get_dtype<TestStruct>());
  // uint32_t dtype_t_ptr = get_dtype<TestStruct*>();
  auto dtype_t_ptr = get_dtype<decltype((TestStruct*)0)>();
  ASSERT_TRUE(dtype_t_ptr.IsPtr());
  ASSERT_TRUE(dtype_t_ptr.IsSameFundamentalType(dtype_t));

  auto dtype_vec = get_dtype<std::vector<int>>();
  ASSERT_EQ(dtype_vec, get_dtype<std::vector<int>>());
  ASSERT_TRUE(dtype_vec.IsVector());
  ASSERT_TRUE(dtype_vec.IsSameFundamentalType(get_dtype<int>()));
  auto dtype_vec_ptr = get_dtype<decltype((std::vector<int>*)0)>();
  ASSERT_TRUE(dtype_vec_ptr.IsPtr());
  ASSERT_TRUE(dtype_vec_ptr.IsSameFundamentalType(dtype_vec));

  auto pair_dtype = get_dtype<std::pair<int, float>>();
  ASSERT_TRUE(pair_dtype.IsTuple());
  auto extract_types = pair_dtype.ExtractTupleDtypes();
  ASSERT_TRUE(extract_types.size() == 2);
  ASSERT_TRUE(extract_types[0] == DATA_I32);
  ASSERT_TRUE(extract_types[1] == DATA_F32);

  auto tuple_dtype = get_dtype<std::tuple<int, float, int64_t>>();
  extract_types = tuple_dtype.ExtractTupleDtypes();
  ASSERT_TRUE(extract_types.size() == 3);
  ASSERT_TRUE(extract_types[0] == DATA_I32);
  ASSERT_TRUE(extract_types[1] == DATA_F32);
  ASSERT_TRUE(extract_types[2] == DATA_I64);
}
namespace test {
namespace internal {
struct TestStruct {};
}  // namespace internal
}  // namespace test
TEST(DTypeFactory, simple) {
  DTypeFactory::Add<int>();
  DTypeFactory::Add<float>();
  DTypeFactory::Add<int64_t>();
  DTypeFactory::Add<::test::internal::TestStruct>();
}

TEST(DTypeFactory, cxxabi) {
  DTypeFactory::Add<int>();
  DTypeFactory::Add<float>();
  DTypeFactory::Add<int64_t>();
  DTypeFactory::Add<::test::internal::TestStruct>();

  const char* name = typeid(std::vector<::test::internal::TestStruct*>).name();
  RUDF_INFO("{}", name);
  char* c_demangled;
  int status = 0;
  c_demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  RUDF_INFO("{}/{}", status, c_demangled);
  free(c_demangled);
}