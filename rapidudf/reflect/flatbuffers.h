/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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

#pragma once
#include <type_traits>
#include "flatbuffers/flatbuffers.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/type_traits.h"
#include "rapidudf/reflect/reflect.h"
namespace rapidudf {

template <typename T>
struct FBSVectorHelper {
  static typename T::value_type Get(const T* fbs_vec, uint32_t i) {
    if (nullptr == fbs_vec) {
      THROW_NULL_POINTER_ERR("null fbs vector");
    }
    if (i >= fbs_vec->size()) {
      return {};
    }
    return fbs_vec->Get(i);
  }
  static size_t Size(const T* fbs_vec) {
    if (nullptr == fbs_vec) {
      return 0;
    }
    return fbs_vec->size();
  }
};

template <typename T>
void try_register_fbs_vector_member_funcs() {
  using remove_ptr_t = std::remove_pointer_t<T>;
  using remove_cv_t = std::remove_cv_t<remove_ptr_t>;
  if constexpr (is_specialization<remove_cv_t, flatbuffers::Vector>::value) {
    Reflect::AddStructMethodAccessor("get", &FBSVectorHelper<remove_cv_t>::Get);
    Reflect::AddStructMethodAccessor("size", &FBSVectorHelper<remove_cv_t>::Size);
  }
}

}  // namespace rapidudf
