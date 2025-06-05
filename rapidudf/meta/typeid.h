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

namespace rapidudf {
class TypeID {
 public:
  template <typename T>
  static uint32_t Get() {
    static uint32_t id = nextTypeId();
    return id;
  }

 private:
  static uint32_t nextTypeId() {
    static std::atomic<uint32_t> type_id_seed = {98765};
    return type_id_seed.fetch_add(1);
  }
};
}  // namespace rapidudf