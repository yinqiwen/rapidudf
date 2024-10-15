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
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rapidudf/reflect/struct.h"
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
  static T from(arg_type v) { return *v; }
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
      THROW_NULL_POINTER_ERR("null vector");
    }
    return STLArgType<value_type>::value(v->at(i));
  }
  static int find(VEC* v, arg_type_t val) {
    if (nullptr == v) {
      return -1;
    }
    for (size_t i = 0; i < v->size(); i++) {
      auto element_v = STLArgType<value_type>::value(v->at(i));
      if (element_v == val) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
  static bool contains(VEC* v, arg_type_t val) {
    int idx = find(v, val);
    return idx > -1;
  }
  static void add(VEC* vec, arg_type_t val) { vec->emplace_back(STLArgType<value_type>::from(val)); }
  static void set(VEC* vec, size_t i, arg_type_t val) {
    if (nullptr == vec) {
      THROW_NULL_POINTER_ERR("null vector");
    }
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
  static void Init() { RUDF_STRUCT_HELPER_METHODS_BIND(VectorHelper<VEC>, get, set, add, size, find, contains) }
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
      THROW_NULL_POINTER_ERR("null map");
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

template <typename T>
void register_stl_collection_member_funcs() {
  using remove_ptr_t = std::remove_pointer_t<T>;
  using remove_cv_t = std::remove_cv_t<remove_ptr_t>;
  if constexpr (is_specialization<remove_cv_t, std::vector>::value) {
    VectorHelper<T>::Init();
  } else if constexpr (is_specialization<remove_cv_t, std::map>::value ||
                       is_specialization<remove_cv_t, std::unordered_map>::value) {
    MapHelper<T>::Init();
  } else if constexpr (is_specialization<remove_cv_t, std::set>::value ||
                       is_specialization<remove_cv_t, std::unordered_set>::value) {
    SetHelper<T>::Init();
  }
}

}  // namespace reflect
}  // namespace rapidudf