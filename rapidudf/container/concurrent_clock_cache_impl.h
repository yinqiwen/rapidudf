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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include "rapidudf/common/allign.h"
#include "rapidudf/container/concurrent_clock_cache.h"

namespace rapidudf {

namespace details {
namespace concurrent_clock_cache {
class OverflowBucket;
class Bucket {
 public:
  static constexpr size_t SIZE = 64;
  static constexpr size_t BUCKET_MASK = 0xF;
  static constexpr size_t BUCKET_MASK_BITS = 4;
  static constexpr size_t FINGERPRINT_MASK = 0x7F;
  static constexpr size_t FINGERPRINT_MASK_BITS = 7;
  struct VisitFlags {
    uint8_t max_clock_k = 3;
    bool update_clock = false;
    bool acquire_slot = false;
  };

  struct Meta {};

  // 元素已被淘汰
  static constexpr uint8_t EVICITED_CONTROL = static_cast<uint8_t>(0x82);
  // 元素正在写入中
  static constexpr uint8_t BUSY_CONTROL = static_cast<uint8_t>(0x81);
  // 可用未知当前也没有写入过元素
  static constexpr uint8_t EMPTY_CONTROL = static_cast<uint8_t>(0x80);

  using VisitFunc = std::function<bool(uint32_t, bool)>;

  Bucket();

  void Find(uint32_t bucket_idx, uint8_t check, VisitFunc&& func, VisitFlags flags);
  void AcquireEmpty(uint32_t bucket_idx, VisitFunc&& func);

  void AddOverflowBucket(std::unique_ptr<OverflowBucket>&& bucket);
  OverflowBucket* GetLastOverflowBucket();

 private:
  uint64_t DoFind(uint8_t check) const;
  int DoEvict(uint32_t from_offset);
  void IncClockCounter(size_t offset, int8_t max_k);
  bool AcquireEmptySlot(size_t offset);
  void SetFingerprint(size_t offset, uint8_t fingerprint);
  void FindOverflow(uint8_t check, VisitFunc&& func, VisitFlags flags);

  uint8_t controls_[SIZE];
  int8_t clock_counters_[SIZE];
  std::unique_ptr<OverflowBucket> overflow_;

  friend class OverflowBucket;
};

class alignas(64) SynchronizedBucket : public Bucket {
 public:
  SynchronizedBucket() = default;
  ~SynchronizedBucket() = default;

 private:
  std::mutex lock_;
};

class OverflowBucket {
 public:
  OverflowBucket();
  bool IsFull() const { return offsets_[Bucket::SIZE - 1] > 0; }

  bool Add(uint32_t offset);

 private:
  void Find(uint8_t check, Bucket::VisitFunc&& func, Bucket::VisitFlags flags);

  Bucket bucket_;
  uint32_t offsets_[Bucket::SIZE];
  friend class Bucket;
};
}  // namespace concurrent_clock_cache
}  // namespace details
template <class K, class V, class Hash, class Eq>
ConcurrentClockCache<K, V, Hash, Eq>::ConcurrentClockCache(const ClockCacheOptions& opts) {
  options_ = opts;
  if (options_.load_factor >= kMaxLoadFactor) {
    options_.load_factor = kMaxLoadFactor;
  }
  if (options_.load_factor <= kMinLoadFactor) {
    options_.load_factor = kMinLoadFactor;
  }
  if (options_.max_size == 0) {
    options_.max_size = kDefaultMaxSize;
  }
  size_t max_size = static_cast<size_t>(options_.max_size / options_.load_factor);
  max_size = align_to<size_t>(max_size, details::concurrent_clock_cache::Bucket::SIZE);
  bucket_count_ = max_size / details::concurrent_clock_cache::Bucket::SIZE;
  bucket_mask_ = bucket_count_ - 1;

  flat_values_.resize(max_size);
  buckets_ = new uint8_t[sizeof(details::concurrent_clock_cache::SynchronizedBucket) * bucket_count_];
  for (size_t i = 0; i < bucket_count_; i++) {
    uint8_t* p = buckets_ + sizeof(details::concurrent_clock_cache::SynchronizedBucket) * i;
    new (p) details::concurrent_clock_cache::SynchronizedBucket;
  }
}
template <class K, class V, class Hash, class Eq>
ConcurrentClockCache<K, V, Hash, Eq>::~ConcurrentClockCache() {
  using SynchronizedBucket = details::concurrent_clock_cache::SynchronizedBucket;
  for (size_t i = 0; i < bucket_count_; i++) {
    SynchronizedBucket* p = reinterpret_cast<SynchronizedBucket*>(buckets_ + sizeof(SynchronizedBucket) * i);
    p->~SynchronizedBucket();
  }
  delete[] buckets_;
}

template <class K, class V, class Hash, class Eq>
uint32_t ConcurrentClockCache<K, V, Hash, Eq>::AddOverflowElement() {
  std::lock_guard<std::mutex> guard(overlow_mutex_);
  overflow_values_.emplace_back(value_type{});
  return overflow_values_.size();
}

template <class K, class V, class Hash, class Eq>
std::shared_ptr<V> ConcurrentClockCache<K, V, Hash, Eq>::Get(const key_type& key) {
  auto hash = hasher()(key);
  uint8_t fingerprint = hash & details::concurrent_clock_cache::Bucket::FINGERPRINT_MASK;
  auto bucket_index = (hash >> details::concurrent_clock_cache::Bucket::FINGERPRINT_MASK_BITS) & bucket_mask_;
  details::concurrent_clock_cache::SynchronizedBucket& bucket =
      reinterpret_cast<details::concurrent_clock_cache::SynchronizedBucket*>(buckets_)[bucket_index];

  std::shared_ptr<V> ret_val;
  bucket.Find(static_cast<uint32_t>(bucket_index), fingerprint, [&](uint32_t offset, bool is_overflow) -> bool {
    const key_type* exist_key = nullptr;
    value_type* exist_value = nullptr;
    if (!is_overflow) {
      exist_key = &flat_values_[offset].first;
      exist_value = &flat_values_[offset].second;
    } else {
      exist_key = &overflow_values_[offset].first;
      exist_value = &overflow_values_[offset].second;
    }
    if (equal()(*exist_key, key)) {
      ret_val = exist_value->load();
      return true;
    }
    return false;
  });
  return ret_val;
}
template <class K, class V, class Hash, class Eq>
std::shared_ptr<V> ConcurrentClockCache<K, V, Hash, Eq>::Put(const key_type& key, value_type&& val, bool overwrite) {
  auto hash = hasher()(key);
  uint8_t fingerprint = hash & details::concurrent_clock_cache::Bucket::FINGERPRINT_MASK;
  auto bucket_index = (hash >> details::concurrent_clock_cache::Bucket::FINGERPRINT_MASK_BITS) & bucket_mask_;
  details::concurrent_clock_cache::SynchronizedBucket& bucket =
      reinterpret_cast<details::concurrent_clock_cache::SynchronizedBucket*>(buckets_)[bucket_index];
  bool found = false;
  std::shared_ptr<V> ret_val;
  bucket.Find(static_cast<uint32_t>(bucket_index), fingerprint, [&](uint32_t offset, bool is_overflow) -> bool {
    const key_type* exist_key = nullptr;
    value_type* old_exist_value = nullptr;
    if (!is_overflow) {
      exist_key = &flat_values_[offset].first;
      old_exist_value = &flat_values_[offset].second;
    } else {
      exist_key = &overflow_values_[offset].first;
      old_exist_value = &overflow_values_[offset].second;
    }
    if (equal()(*exist_key, key)) {
      found = true;
      if (overwrite) {
        old_exist_value->store(std::make_shared<V>(std::move(val)));
      } else {
        ret_val = old_exist_value->load();
      }
      return true;
    }
    return false;
  });
  if (found) {
    return overwrite ? nullptr : ret_val;
  }
  bool found_empty = false;
  bucket.AcquireEmpty(static_cast<uint32_t>(bucket_index), [&](uint32_t offset, bool is_overflow) -> bool {
    value_type* memory = nullptr;
    if (!is_overflow) {
      memory = &flat_values_[offset];
    } else {
      memory = &overflow_values_[offset];
    }
    memory->first = key;
    memory->second.store(std::make_shared<V>(std::move(val)));
    found_empty = true;
    size_++;
    return true;
  });
  if (found_empty) {
    return nullptr;
  }

  uint32_t overflow_element_offset = AddOverflowElement();
  overflow_values_[overflow_element_offset].first = key;
  overflow_values_[overflow_element_offset].second.store(std::make_shared<V>(std::move(val)));

  std::lock_guard<std::mutex> guard();
  details::concurrent_clock_cache::OverflowBucket* last_overflow_bucket = bucket.GetLastOverflowBucket();
  if (last_overflow_bucket == nullptr || last_overflow_bucket->IsFull()) {
    auto new_overflow_bucket = std::make_unique<details::concurrent_clock_cache::OverflowBucket>();
    last_overflow_bucket = new_overflow_bucket.get();
    bucket.AddOverflowBucket(std::move(new_overflow_bucket));
  }
  last_overflow_bucket->Add(overflow_element_offset);
  return nullptr;
}
}  // namespace rapidudf