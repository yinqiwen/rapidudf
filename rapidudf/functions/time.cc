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

#include <sys/time.h>
#include <time.h>
#include <cstdint>

#include "rapidudf/meta/function.h"

namespace rapidudf {

namespace functions {

static uint64_t now_us() {
  struct timeval tv;
  uint64_t ust;
  gettimeofday(&tv, nullptr);
  ust = ((int64_t)tv.tv_sec) * 1000000;
  ust += tv.tv_usec;
  return ust;
}
static uint64_t now_ms() { return now_us() / 1000; }
static uint64_t now_s() { return now_us() / 1000000; }

void init_builtin_time_funcs() {
  RUDF_FUNC_REGISTER(now_us);
  RUDF_FUNC_REGISTER(now_ms);
  RUDF_FUNC_REGISTER(now_s);
}
}  // namespace functions
}  // namespace rapidudf