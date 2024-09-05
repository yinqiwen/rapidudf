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
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rapidudf/meta/type_traits.h"
#include "rapidudf/reflect/reflect.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace reflect {
template <typename T, typename Enable = void>
struct STLGetterReturnType {
  using return_type = const T*;
  static return_type value(const T& v) { return &v; }
  static return_type default_value() {
    static T empty;
    return &empty;
  }
};
template <typename T>
struct STLGetterReturnType<
    T, typename std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T> ||
                               std::is_same_v<std::string_view, T> || std::is_same_v<StringView, T>>::type> {
  using return_type = T;
  static return_type value(const T& v) { return v; }
  static return_type default_value() { return {}; }
};
template <>
struct STLGetterReturnType<std::string> {
  using return_type = std::string_view;
  static return_type value(const std::string& v) { return v; }
};
template <>
struct STLGetterReturnType<std::string_view> {
  using return_type = std::string_view;
  static return_type value(const std::string& v) { return v; }
};

template <typename T>
struct VectorHelper {
  using return_type_t = typename STLGetterReturnType<T>::return_type;
  static return_type_t get(std::vector<T>* v, size_t i) {
    if (nullptr == v) {
      return STLGetterReturnType<T>::default_value();
    }
    return STLGetterReturnType<T>::value(v->at(i));
  }
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
};

template <typename Set>
struct SetHelper {
  using key_type = typename Set::key_type;
  static bool contains(Set* v, key_type val) {
    if (nullptr == v) {
      return false;
    }
    return v->find(val) != v->end();
  }
  //   static bool insert(Set* vec, key_type val) {
  //     if (nullptr == vec) {
  //       return false;
  //     }
  //     return vec->insert(val).second;
  //   }
  static size_t size(Set* vec) {
    if (nullptr == vec) {
      return 0;
    }
    return vec->size();
  }
};

template <template <class, class> class Map, class K, class V>
struct MapHelper {
  static bool contains(Map<K, V>* v, K key) {
    if (nullptr == v) {
      return false;
    }
    return v->find(key) != v->end();
  }
  static V get(Map<K, V>* v, K key) {
    if (nullptr == v) {
      return {};
    }
    auto found = v->find(key);
    if (found == v->end()) {
      return {};
    }
    return found->second;
  }
  static bool insert(Map<K, V>* map, K key, V val) {
    if (nullptr == map) {
      return false;
    }
    return map->emplace(key, val).second;
  }
  static size_t size(Map<K, V>* map) {
    if (nullptr == map) {
      return 0;
    }
    return map->size();
  }
};

template <typename T>
void try_register_stl_collection_member_funcs() {
  using remove_ptr_t = std::remove_pointer_t<T>;
  using remove_reference_t = std::remove_reference_t<remove_ptr_t>;
  using remove_cv_t = std::remove_cv_t<remove_reference_t>;
  if constexpr (is_specialization<remove_cv_t, std::vector>::value) {
    using key_type = typename remove_cv_t::value_type;
    Reflect::AddStructMethodAccessor("get", &VectorHelper<key_type>::get);
    Reflect::AddStructMethodAccessor("size", &VectorHelper<key_type>::size);
  } else if constexpr (is_specialization<remove_cv_t, std::set>::value) {
    using key_type = typename remove_cv_t::key_type;
    Reflect::AddStructMethodAccessor("contains", &SetHelper<std::set<key_type>>::contains);
    Reflect::AddStructMethodAccessor("size", &SetHelper<std::set<key_type>>::size);
  }
}

}  // namespace reflect
}  // namespace rapidudf