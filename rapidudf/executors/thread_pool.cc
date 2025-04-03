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
#include "rapidudf/executors/thread_pool.h"
#include <memory>
#include <mutex>
namespace rapidudf {
static std::once_flag g_thread_pool_init_flag;
static std::shared_ptr<ThreadPool> g_thread_pool;
std::shared_ptr<ThreadPool> get_global_thread_pool(size_t num) {
  std::call_once(g_thread_pool_init_flag, [num]() { init_global_thread_pool(num); });
  return g_thread_pool;
}
void init_global_thread_pool(size_t num, typename ThreadPool::AsyncExecutor&& closure_executor) {
  if (!g_thread_pool) {
    g_thread_pool = std::make_shared<ThreadPool>(std::min(num, ThreadPool::MaxThreads()), std::move(closure_executor));
  }
}
}  // namespace rapidudf