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

#include "rapidudf/reflect/struct.h"
#include "rapidudf/types/simd_vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace reflect {
template <typename T, typename Enable = void>
struct STLArgType {
  using arg_type = const T*;
  static arg_type value(const T& v) { return &v; }
  static arg_type default_value() {
    static T empty;
    return &empty;
  }
};
template <typename T>
struct STLArgType<T,
                  typename std::enable_if<std::is_integral_v<T> || std::is_floating_point_v<T> ||
                                          std::is_pointer_v<T> || std::is_same_v<std::string_view, T> ||
                                          std::is_same_v<StringView, T> || std::is_same_v<std::string_view, T>>::type> {
  using arg_type = T;
  static arg_type value(const T& v) { return v; }
  static arg_type default_value() { return {}; }
  static T from(arg_type v) { return v; }
};
template <>
struct STLArgType<std::string> {
  using arg_type = StringView;
  static arg_type value(const std::string& v) { return StringView(v); }
  static std::string from(arg_type v) { return v.str(); }
  static arg_type default_value() { return ""; }
};

template <typename VEC>
struct VectorHelper {
  using value_type = typename VEC::value_type;
  using arg_type_t = typename STLArgType<value_type>::arg_type;
  static arg_type_t get(VEC* v, size_t i) {
    if (nullptr == v) {
      return STLArgType<value_type>::default_value();
    }
    return STLArgType<value_type>::value(v->at(i));
  }
  static void add(VEC* vec, arg_type_t val) { vec->emplace_back(STLArgType<value_type>::from(val)); }
  static void set(VEC* vec, size_t i, arg_type_t val) {
    if (vec->size() > i) {
      vec->at(i) = STLArgType<value_type>::from(val);
    }
  }
  static size_t size(VEC* vec) {
    if (nullptr == vec) {
      return 0;
    }
    return vec->size();
  }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(VectorHelper<VEC>, get, set, add, size) }
};

template <typename T>
using StdVectorHelper = VectorHelper<std::vector<T>>;

template <typename Set>
struct SetHelper {
  using key_type = typename Set::key_type;
  using arg_type_t = typename STLArgType<key_type>::arg_type;
  static bool contains(Set* v, arg_type_t val) {
    if (nullptr == v) {
      return false;
    }
    return v->find(STLArgType<key_type>::from(val)) != v->end();
  }
  static bool insert(Set* vec, arg_type_t val) {
    if (nullptr == vec) {
      return false;
    }
    return vec->insert(STLArgType<key_type>::from(val)).second;
  }
  static size_t size(Set* vec) {
    if (nullptr == vec) {
      return 0;
    }
    return vec->size();
  }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(SetHelper<Set>, contains, insert, size) }
};

template <typename T>
using StdSetHelper = SetHelper<std::set<T>>;

template <typename T>
using StdUnorderedSetHelper = SetHelper<std::unordered_set<T>>;

template <typename Map>
struct MapHelper {
  using key_type = typename Map::key_type;
  using value_type = typename Map::mapped_type;
  using arg_key_type_t = typename STLArgType<key_type>::arg_type;
  using arg_value_type_t = typename STLArgType<value_type>::arg_type;
  static bool contains(Map* v, arg_key_type_t key) {
    if (nullptr == v) {
      return false;
    }
    return v->find(STLArgType<key_type>::from(key)) != v->end();
  }
  static arg_value_type_t get(Map* v, arg_key_type_t key) {
    if (nullptr == v) {
      return {};
    }
    auto found = v->find(STLArgType<key_type>::from(key));
    if (found == v->end()) {
      return {};
    }
    return STLArgType<value_type>::value(found->second);
  }
  static bool insert(Map* map, arg_key_type_t key, arg_value_type_t val) {
    if (nullptr == map) {
      return false;
    }
    return map->emplace(STLArgType<key_type>::from(key), STLArgType<value_type>::from(val)).second;
  }
  static size_t size(Map* map) {
    if (nullptr == map) {
      return 0;
    }
    return map->size();
  }
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(MapHelper<Map>, contains, get, insert, size) }
};

template <typename K, typename V>
using StdMapHelper = MapHelper<std::map<K, V>>;

template <typename K, typename V>
using StdUnorderedMapHelper = MapHelper<std::unordered_map<K, V>>;

}  // namespace reflect
}  // namespace rapidudf