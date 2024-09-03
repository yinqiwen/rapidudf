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
#include "rapidudf/reflect/stl_access.h"
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <vector>
#include "rapidudf/reflect/struct_access.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
template <typename T>
struct VectorReflectHelper {
  static T get(std::vector<T>* v, size_t i) { return v->at(i); }
  static void add(std::vector<T>* vec, T val) { vec->emplace_back(val); }
  static void set(std::vector<T>* vec, size_t i, T val) {
    if (vec->size() > i) {
      vec->at(i) = val;
    }
  }
  static size_t size(std::vector<T>* vec) {
    if (nullptr == vec) {
      return 0;
    }
    return vec->size();
  }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(VectorReflectHelper<T>, get, set, add, size) }
};
template <typename T>
struct SimdVectorReflectHelper {
  static T get(simd::Vector<T> v, size_t i) { return v[i]; }
  static size_t size(simd::Vector<T> v) { return v.Size(); }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(SimdVectorReflectHelper<T>, get, size) }
};

template <typename T>
struct SetReflectHelper {
  static bool contains(std::set<T>* v, T val) {
    if (nullptr == v) {
      return false;
    }
    return v->find(val) != v->end();
  }
  static bool insert(std::set<T>* vec, T val) {
    if (nullptr == vec) {
      return false;
    }
    return vec->insert(val).second;
  }
  static size_t size(std::set<T>* vec) {
    if (nullptr == vec) {
      return 0;
    }
    return vec->size();
  }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(SetReflectHelper<T>, contains, insert, size) }
};

#define RUDF_STL_REFLECT_HELPER_INIT(r, STL_HELPER, i, TYPE) STL_HELPER<TYPE>::Init();
#define RUDF_STL_REFLECT_HELPER(STL_HELPER, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(RUDF_STL_REFLECT_HELPER_INIT, STL_HELPER, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

void init_stl_reflect_access() {
  RUDF_STL_REFLECT_HELPER(VectorReflectHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                          float, double, std::string_view)
  RUDF_STL_REFLECT_HELPER(SetReflectHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                          float, double, std::string_view)
  RUDF_STL_REFLECT_HELPER(SimdVectorReflectHelper, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                          int64_t, float, double, simd::Bit, StringView)
}
}  // namespace rapidudf