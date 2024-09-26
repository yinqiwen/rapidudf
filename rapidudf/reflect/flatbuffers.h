/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
