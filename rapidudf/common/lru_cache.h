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

#pragma once
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include "absl/container/flat_hash_map.h"
#include "boost/thread/tss.hpp"

// copy & modified from boost
namespace rapidudf {

// a cache which evicts the least recently used item when it is full
template <class Key, class Value, class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq = absl::container_internal::hash_default_eq<Key>>
class lru_cache {
 public:
  typedef Key key_type;
  typedef Value value_type;
  typedef std::list<key_type> list_type;
  typedef absl::flat_hash_map<key_type, std::pair<value_type, typename list_type::iterator>, Hash, Eq> map_type;

  lru_cache(size_t capacity) : m_capacity(capacity) {}

  ~lru_cache() {}

  size_t size() const { return m_map.size(); }

  size_t capacity() const { return m_capacity; }

  void reset_capacity(size_t n) { m_capacity = n; }

  bool empty() const { return m_map.empty(); }

  bool contains(const key_type& key) { return m_map.find(key) != m_map.end(); }

  void insert(const key_type& key, const value_type& value) {
    typename map_type::iterator i = m_map.find(key);
    if (i == m_map.end()) {
      // insert item into the cache, but first check if it is full
      if (size() >= m_capacity) {
        // cache is full, evict the least recently used item
        evict();
      }

      // insert the new item
      m_list.push_front(key);
      m_map[key] = std::make_pair(value, m_list.begin());
    }
  }

  void erase(const key_type& key) {
    // lookup value in the cache
    typename map_type::iterator i = m_map.find(key);
    if (i == m_map.end()) {
      // value not in cache
      return;
    }

    // return the value, but first update its place in the most
    // recently used list
    typename list_type::iterator j = i->second.second;
    m_list.erase(j);
    m_map.erase(i);
  }

  std::optional<value_type> get(const key_type& key) {
    // lookup value in the cache
    typename map_type::iterator i = m_map.find(key);
    if (i == m_map.end()) {
      // value not in cache
      return {};
    }

    // return the value, but first update its place in the most
    // recently used list
    typename list_type::iterator j = i->second.second;
    if (j != m_list.begin()) {
      // move item to the front of the most recently used list
      m_list.erase(j);
      m_list.push_front(key);

      // update iterator in map
      j = m_list.begin();
      const value_type& value = i->second.first;
      m_map[key] = std::make_pair(value, j);

      // return the value
      return value;
    } else {
      // the item is already at the front of the most recently
      // used list so just return it
      return i->second.first;
    }
  }

  void clear() {
    m_map.clear();
    m_list.clear();
  }

 private:
  void evict() {
    // evict item from the end of most recently used list
    typename list_type::iterator i = --m_list.end();
    m_map.erase(*i);
    m_list.erase(i);
  }

 private:
  map_type m_map;
  list_type m_list;
  size_t m_capacity;
};

template <class Key, class Value, class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq = absl::container_internal::hash_default_eq<Key>>
class ThreadCachedLRU {
 public:
  using key_type = Key;
  using value_type = Value;
  using lru_type = lru_cache<Key, Value, Hash, Eq>;
  static constexpr size_t kDefaultCacapacity = 1024;
  ThreadCachedLRU(size_t capacity) : capacity_(capacity) {}
  bool Contains(const key_type& key) { return GetLRU()->contains(key); }

  void Insert(const key_type& key, const value_type& value) { GetLRU()->insert(key, value); }
  std::optional<value_type> Get(const key_type& key) { return GetLRU()->get(key); }
  void Erase(const key_type& key) { GetLRU()->erase(key); }

  value_type GetOrInsert(const key_type& key, const std::function<value_type()>& func) {
    auto result = Get(key);
    if (result.has_value()) {
      return *result;
    }
    auto val = func();
    Insert(key, val);
    return val;
  }

 private:
  lru_type* GetLRU() {
    lru_type* lru = lru_.get();
    if (lru == nullptr) {
      lru = new lru_type(capacity_);
      lru_.reset(lru);
    }
    return lru;
  }
  boost::thread_specific_ptr<lru_type> lru_;
  size_t capacity_;
};

template <class Key, class Value, class Hash = absl::container_internal::hash_default_hash<Key>,
          class Eq = absl::container_internal::hash_default_eq<Key>>
std::shared_ptr<ThreadCachedLRU<Key, Value, Hash, Eq>> get_global_lru_cache(
    size_t num = ThreadCachedLRU<Key, Value, Hash, Eq>::kDefaultCacapacity) {
  static std::once_flag g_lru_init_flag;
  static std::shared_ptr<ThreadCachedLRU<Key, Value, Hash, Eq>> g_lru;
  std::call_once(g_lru_init_flag, [num]() { g_lru = std::make_shared<ThreadCachedLRU<Key, Value, Hash, Eq>>(num); });
  return g_lru;
}

}  // namespace rapidudf