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
#include <functional>
#include <memory>
#include "boost/asio/post.hpp"
#include "boost/asio/thread_pool.hpp"
#include "hwy/contrib/thread_pool/thread_pool.h"
namespace rapidudf {

class ThreadPool {
 public:
  static constexpr size_t kDefaultThreadNum = 8;
  using AnyClosure = std::function<void(void)>;
  using AsyncExecutor = std::function<void(AnyClosure&&)>;
  static size_t MaxThreads() { return hwy::ThreadPool::MaxThreads(); };
  explicit ThreadPool(size_t num_threads, AsyncExecutor&& async_executor = {})
      : async_executor_(std::move(async_executor)), num_workers_(num_threads) {
    if (num_workers_ == 0) {
      num_workers_ = kDefaultThreadNum;
    }
    if (!async_executor_) {
      thread_pool_ = std::make_unique<boost::asio::thread_pool>(num_workers_);
    }
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  /**
   * 'Run' wait for all closure done
   */
  template <class Closure>
  void Run(uint64_t begin, uint64_t end, const Closure& closure, size_t num_workers = 0) {
    hwy::PoolBarrier barrier;
    barrier.Reset();
    if (HWY_LIKELY(num_workers == 0)) {
      num_workers = num_workers_;
    } else {
      num_workers = std::min(num_workers, num_workers_);
    }
    size_t launched_worker = Plan(begin, end, num_workers, closure, barrier);
    if (HWY_LIKELY(launched_worker > 0)) {
      barrier.WaitAll(launched_worker);
    }
  }

 private:
  template <class Closure>
  size_t Plan(uint64_t begin, uint64_t end, size_t num_workers, const Closure& closure, hwy::PoolBarrier& barrier) {
    // If there are no tasks, we are done.
    HWY_DASSERT(begin <= end);
    const size_t num_tasks = static_cast<size_t>(end - begin);
    if (HWY_UNLIKELY(num_tasks == 0)) return 0;

    // If there are no workers, run all tasks already on the main thread without
    // the overhead of planning.
    if (HWY_UNLIKELY(num_workers <= 1)) {
      for (uint64_t task = begin; task < end; ++task) {
        closure(task);
      }
      return 0;
    }

    // Assigning all remainders to the last thread causes imbalance. We instead
    // give one more to each thread whose index is less.
    const size_t remainder = num_tasks % num_workers;
    const size_t min_tasks = num_tasks / num_workers;

    size_t total_lanched_worker = 0;
    uint64_t task = begin;
    for (size_t thread = 0; thread < num_workers; ++thread) {
      if (task >= end) {
        break;
      }
      const uint64_t my_start = task;
      const uint64_t my_end = task + min_tasks + (thread < remainder);
      auto task_f = [my_start, my_end, thread, &closure, &barrier]() {
        for (size_t i = my_start; i < my_end; i++) {
          closure(i);
        }
        barrier.WorkerArrive(thread);
      };
      if (thread_pool_) {
        boost::asio::post(*thread_pool_, task_f);
      } else {
        async_executor_(std::move(task_f));
      }
      task = my_end;
      total_lanched_worker++;
    }
    HWY_DASSERT(task == end);
    return total_lanched_worker;
  }

  std::unique_ptr<boost::asio::thread_pool> thread_pool_;
  AsyncExecutor async_executor_;
  size_t num_workers_;
};
void init_global_thread_pool(size_t num = ThreadPool::kDefaultThreadNum,
                             typename ThreadPool::AsyncExecutor&& closure_executor = {});
std::shared_ptr<ThreadPool> get_global_thread_pool(size_t num = ThreadPool::kDefaultThreadNum);

}  // namespace rapidudf