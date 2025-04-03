/*
 * Copyright (c) 2025 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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

#include <deque>
#include <forward_list>
#include <list>
#include <type_traits>
#include <vector>
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "rapidudf/memory/arena.h"
namespace rapidudf {
namespace arena {
template <typename T, typename A = ThreadSafeArenaAllocator<T>>
using Vector = ::std::vector<T, A>;

template <typename T, typename A = ThreadSafeArenaAllocator<T>>
using List = ::std::list<T, A>;

template <typename T, typename A = ThreadSafeArenaAllocator<T>>
using ForwardList = ::std::forward_list<T, A>;

template <typename T, typename A = ThreadSafeArenaAllocator<T>>
using Deque = ::std::deque<T, A>;

template <class K, class V, class Hash = absl::container_internal::hash_default_hash<K>,
          class Eq = absl::container_internal::hash_default_eq<K>,
          class A = ThreadSafeArenaAllocator<std::pair<const K, V>>>
using HashMap = absl::flat_hash_map<K, V, Hash, Eq, A>;

template <class K, class Hash = absl::container_internal::hash_default_hash<K>,
          class Eq = absl::container_internal::hash_default_eq<K>, class A = ThreadSafeArenaAllocator<K>>
using HashSet = absl::flat_hash_set<K, Hash, Eq, A>;

template <typename Key, typename Value, typename Compare = std::less<Key>,
          typename A = ThreadSafeArenaAllocator<std::pair<const Key, Value>>>
using BTreeMap = absl::btree_map<Key, Value, Compare, A>;

template <typename Key, typename Compare = std::less<Key>, class A = ThreadSafeArenaAllocator<Key>>
using BTreeSet = absl::btree_set<Key, Compare, A>;

template <class K, class V, class Hash = absl::container_internal::hash_default_hash<K>,
          class Eq = absl::container_internal::hash_default_eq<K>,
          class A = ThreadSafeArenaAllocator<std::pair<const K, V>>>
using NodeHashMap = absl::node_hash_map<K, V, Hash, Eq, A>;

template <class K, class Hash = absl::container_internal::hash_default_hash<K>,
          class Eq = absl::container_internal::hash_default_eq<K>, class A = ThreadSafeArenaAllocator<K>>
using NodeHashSet = absl::node_hash_set<K, Hash, Eq, A>;

}  // namespace arena
}  // namespace rapidudf

namespace std {
template <typename T, typename A>
struct is_trivially_destructible<::rapidudf::arena::Vector<T, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<T>;
};

template <typename T, typename A>
struct is_trivially_destructible<::rapidudf::arena::List<T, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<T>;
};

template <typename T, typename A>
struct is_trivially_destructible<::rapidudf::arena::ForwardList<T, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<T>;
};

template <typename T, typename A>
struct is_trivially_destructible<::rapidudf::arena::Deque<T, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<T>;
};

template <class K, class V, class Hash, class Eq, typename A>
struct is_trivially_destructible<::rapidudf::arena::HashMap<K, V, Hash, Eq, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<K> && rapidudf::is_destructor_disabled_v<V>;
};
template <class K, class V, class Hash, class Eq, typename A>
struct is_trivially_destructible<::rapidudf::arena::NodeHashMap<K, V, Hash, Eq, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<K> && rapidudf::is_destructor_disabled_v<V>;
};

template <class K, class Hash, class Eq, typename A>
struct is_trivially_destructible<::rapidudf::arena::HashSet<K, Hash, Eq, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<K>;
};
template <class K, class Hash, class Eq, typename A>
struct is_trivially_destructible<::rapidudf::arena::NodeHashSet<K, Hash, Eq, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<K>;
};

template <typename Key, typename Value, typename Compare, typename A>
struct is_trivially_destructible<::rapidudf::arena::BTreeMap<Key, Value, Compare, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<Key> && rapidudf::is_destructor_disabled_v<Value>;
};
template <typename Key, typename Compare, typename A>
struct is_trivially_destructible<::rapidudf::arena::BTreeSet<Key, Compare, A>> {
  static constexpr bool value = rapidudf::is_destructor_disabled_v<Key>;
};

}  // namespace std