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

#include "rapidudf/meta/dtype.h"
#include <cxxabi.h>
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "rapidudf/log/log.h"

struct TestStruct {
  int a;
};

using namespace rapidudf;
TEST(DType, ptr) {
  auto ptr_dtype = get_dtype<int*>();
  ASSERT_TRUE(ptr_dtype.IsIntegerPtr());
  auto ptr_ptr_dtype = get_dtype<int**>();
  ASSERT_TRUE(ptr_ptr_dtype.IsPtr());
  ASSERT_FALSE(ptr_ptr_dtype.IsIntegerPtr());
  ASSERT_TRUE(ptr_ptr_dtype.PtrTo().IsIntegerPtr());
  auto ptr_ptr_ptr_dtype = get_dtype<int***>();
  ASSERT_TRUE(ptr_ptr_ptr_dtype.IsPtr());
  ASSERT_FALSE(ptr_ptr_ptr_dtype.IsIntegerPtr());
  ASSERT_TRUE(ptr_ptr_ptr_dtype.PtrTo().PtrTo().IsIntegerPtr());
}

TEST(DType, vector) {
  auto vector_dtype = get_dtype<std::vector<int>>();
  ASSERT_TRUE(vector_dtype.IsVector());
  auto vector2_dtype = get_dtype<std::vector<std::vector<int>>>();
  ASSERT_TRUE(vector2_dtype.IsVector());
  ASSERT_TRUE(vector2_dtype.Elem().IsVector());
  ASSERT_TRUE(vector2_dtype.Elem().Elem().IsInteger());
}

TEST(DType, map) {
  auto map_dtype = get_dtype<std::map<int, int>>();
  ASSERT_TRUE(map_dtype.IsMap());
  auto map_vec_dtype = get_dtype<std::map<int, std::vector<int>>>();
  ASSERT_TRUE(map_vec_dtype.IsMap());
  ASSERT_TRUE(map_vec_dtype.Elem().IsVector());
  ASSERT_TRUE(map_vec_dtype.Elem().Elem().IsInteger());
}

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

  auto tuple_dtype = get_dtype<std::tuple<int, float>>();
  extract_types = tuple_dtype.ExtractTupleDtypes();
  ASSERT_TRUE(extract_types.size() == 2);
  ASSERT_TRUE(extract_types[0] == DATA_I32);
  ASSERT_TRUE(extract_types[1] == DATA_F32);
  // ASSERT_TRUE(extract_types[2] == DATA_I64);
}

TEST(DType, vector_array) {
  auto dtype = get_dtype<Vector<std::array<float, 6>>>();
  ASSERT_TRUE(dtype.IsSimdVector());
  ASSERT_TRUE(dtype.Elem().IsArray(6));
  ASSERT_TRUE(dtype.Elem().Elem().IsF32());
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