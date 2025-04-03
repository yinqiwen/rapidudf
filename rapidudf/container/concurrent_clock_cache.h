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
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "boost/smart_ptr/atomic_shared_ptr.hpp"

namespace rapidudf {

struct ClockCacheOptions {
  size_t segment = 16;
  size_t max_size = 1024 * 1024;
  uint8_t k = 3;
  float load_factor = 0.8;
};

template <class K, class V, class Hash = absl::container_internal::hash_default_hash<K>,
          class Eq = absl::container_internal::hash_default_eq<K>>
class ConcurrentClockCache {
 public:
  using key_type = K;
  using mapped_type = V;
  using value_type = std::pair<K, boost::atomic_shared_ptr<mapped_type>>;
  using hasher = Hash;
  using equal = Eq;
  static constexpr float kMaxLoadFactor = 0.9;
  static constexpr float kMinLoadFactor = 0.1;
  static constexpr size_t kDefaultMaxSize = 1024 * 1024;

  explicit ConcurrentClockCache(const ClockCacheOptions& opts);

  std::shared_ptr<mapped_type> Get(const key_type& k);
  std::shared_ptr<mapped_type> Put(const key_type& k, value_type&& v, bool overwrite = false);
  size_t Size() const { return size_.load(); }

  ~ConcurrentClockCache();

 private:
  void Evict();
  uint32_t AddOverflowElement();

  ClockCacheOptions options_;

  std::vector<value_type> flat_values_;
  std::deque<value_type> overflow_values_;
  std::mutex overlow_mutex_;

  uint8_t* buckets_{nullptr};
  size_t bucket_count_{0};
  size_t bucket_mask_{0};

  ::std::atomic<size_t> size_{0};
  // ::std::atomic<uint64_t> clock_pointer_{0};
};
}  // namespace rapidudf