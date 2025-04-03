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
#include "rapidudf/container/concurrent_clock_cache_impl.h"
#include <sched.h>
#include "rapidudf/functions/simd/bits.h"
#include "rapidudf/functions/simd/vector_misc.h"

namespace rapidudf {
namespace details {
namespace concurrent_clock_cache {

Bucket::Bucket() {
  for (size_t i = 0; i < Bucket::SIZE; i++) {
    controls_[i] = EMPTY_CONTROL;
  }
  memset(clock_counters_, 0, Bucket::SIZE);
}

int Bucket::DoEvict(uint32_t from_offset) {
  int n = 0;
  for (; from_offset < SIZE; from_offset++) {
    int8_t k = clock_counters_[from_offset];
    if (k <= 0) {
      continue;
    }
    int8_t old_value = __sync_val_compare_and_swap(&clock_counters_[from_offset], k, k - 1);
    if (old_value == 1 && k == 1) {
      // destroy

      return 1;
    }
  }
  return n;
}

uint64_t Bucket::DoFind(uint8_t check) const {
  uint64_t mask = 0;
  functions::simd_vector_match<uint8_t, OP_EQUAL>(controls_, SIZE, check, mask);
  return mask;
}
void Bucket::FindOverflow(uint8_t check, VisitFunc&& func, VisitFlags flags) {
  if (overflow_) {
    overflow_->Find(check, std::move(func), flags);
  }
}
void Bucket::IncClockCounter(size_t offset, int8_t max_k) {
  while (1) {
    int8_t k = clock_counters_[offset];
    if (k >= max_k) {
      return;
    }
    bool result = __sync_bool_compare_and_swap(&clock_counters_[offset], k, k + 1);
    if (result) {
      return;
    }
  }
}
bool Bucket::AcquireEmptySlot(size_t offset) {
  while (1) {
    uint8_t old_value = __sync_val_compare_and_swap(&controls_[offset], EMPTY_CONTROL, BUSY_CONTROL);
    if (old_value == BUSY_CONTROL) {
      ::sched_yield();
      continue;
    } else if (old_value == EMPTY_CONTROL) {
      return true;
    } else {
      return false;
    }
  }
}
void Bucket::SetFingerprint(size_t offset, uint8_t fingerprint) {
  while (1) {
    uint8_t old_value = __sync_val_compare_and_swap(&controls_[offset], BUSY_CONTROL, fingerprint);
    if (old_value == BUSY_CONTROL) {
      return;
    }
  }
}

void Bucket::Find(uint32_t bucket_idx, uint8_t fingerprint, VisitFunc&& func, VisitFlags flags) {
  uint64_t mask = DoFind(fingerprint);
  bits::MaskIterator iter(mask);
  while (iter) {
    size_t offset = iter.Advance();
    if (flags.acquire_slot) {
      if (!AcquireEmptySlot(offset)) {
        continue;
      }
      ::std::atomic_thread_fence(::std::memory_order_acquire);
    }

    bool rc = func(bucket_idx * SIZE + offset, true);
    if (rc) {
      if (flags.update_clock) {
        // inc clock counter
        IncClockCounter(offset, flags.max_clock_k);
      }
      if (flags.acquire_slot) {
        ::std::atomic_thread_fence(::std::memory_order_acquire);
        SetFingerprint(offset, fingerprint);
      }
      return;
    }
  }
  FindOverflow(fingerprint, std::move(func), flags);
}
void Bucket::AcquireEmpty(uint32_t bucket_idx, VisitFunc&& func) {
  Find(bucket_idx, EMPTY_CONTROL, std::move(func), VisitFlags{.acquire_slot = true});
}

void Bucket::AddOverflowBucket(std::unique_ptr<OverflowBucket>&& bucket) {
  if (!overflow_) {
    overflow_ = std::move(bucket);
  } else {
    overflow_->bucket_.AddOverflowBucket(std::move(bucket));
  }
}
OverflowBucket* Bucket::GetLastOverflowBucket() {
  if (!overflow_) {
    return nullptr;
  }
  if (!overflow_->bucket_.overflow_) {
    return overflow_.get();
  }
  return overflow_->bucket_.GetLastOverflowBucket();
}

void OverflowBucket::Find(uint8_t fingerprint, Bucket::VisitFunc&& func, Bucket::VisitFlags flags) {
  uint64_t mask = bucket_.DoFind(fingerprint);
  bits::MaskIterator iter(mask);
  while (iter) {
    size_t offset = iter.Advance();
    if (offsets_[offset] == 0) {
      return;
    }
    if (flags.acquire_slot) {
      if (!bucket_.AcquireEmptySlot(offset)) {
        continue;
      }
      ::std::atomic_thread_fence(::std::memory_order_acquire);
    }
    bool rc = func(offsets_[offset] - 1, true);
    if (rc) {
      if (flags.update_clock) {
        // inc clock counter
        bucket_.IncClockCounter(offset, flags.max_clock_k);
      }
      if (flags.acquire_slot) {
        ::std::atomic_thread_fence(::std::memory_order_acquire);
        bucket_.SetFingerprint(offset, fingerprint);
      }
      return;
    }
  }
  bucket_.FindOverflow(fingerprint, std::move(func), flags);
}

OverflowBucket::OverflowBucket() { memset(offsets_, 0, sizeof(uint32_t) * Bucket::SIZE); }

bool OverflowBucket::Add(uint32_t offset) {
  for (size_t i = 0; i < Bucket::SIZE; i++) {
    if (offsets_[i] == 0) {
      offsets_[i] = offset + 1;
      return true;
    }
  }
  return false;
}

}  // namespace concurrent_clock_cache
}  // namespace details
}  // namespace rapidudf