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
#include "rapidudf/exec/eval_engine.h"
namespace rapidudf {
namespace exec {
static const int kDefaultLRUCacheSize = 10000;
EvalCache& get_eval_cache() {
  static EvalCache cache(kDefaultLRUCacheSize);
  return cache;
}
}  // namespace exec
}  // namespace rapidudf