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
#include "rapidudf/log/log.h"
namespace rapidudf {
std::shared_ptr<spdlog::logger> g_default_looger;
void set_default_logger(std::shared_ptr<spdlog::logger> logger) { g_default_looger = logger; }
spdlog::logger* get_default_raw_logger() {
  return g_default_looger ? g_default_looger.get() : spdlog::default_logger_raw();
}

}  // namespace rapidudf