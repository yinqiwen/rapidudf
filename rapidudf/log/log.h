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
#include <memory>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

namespace rapidudf {
extern std::shared_ptr<spdlog::logger> g_default_looger;
void set_default_logger(std::shared_ptr<spdlog::logger> logger);
spdlog::logger* get_default_raw_logger();

}  // namespace rapidudf

#define RUDF_LOG(level, ...)                                                                          \
  do {                                                                                                \
    auto _local_logger_ =                                                                             \
        rapidudf::g_default_looger ? rapidudf::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(level)) {                             \
      SPDLOG_LOGGER_CALL(_local_logger_, level, __VA_ARGS__);                                         \
    }                                                                                                 \
  } while (0)

#define RUDF_TRACE(...) RUDF_LOG(spdlog::level::trace, __VA_ARGS__)
#define RUDF_DEBUG(...) RUDF_LOG(spdlog::level::debug, __VA_ARGS__)
#define RUDF_INFO(...) RUDF_LOG(spdlog::level::info, __VA_ARGS__)
#define RUDF_WARN(...) RUDF_LOG(spdlog::level::warn, __VA_ARGS__)
#define RUDF_ERROR(...) RUDF_LOG(spdlog::level::err, __VA_ARGS__)
#define RUDF_CRITICAL(...) RUDF_LOG(spdlog::level::critical, __VA_ARGS__)

#define RUDF_LOG_EVERY_N(level, n, ...)                                   \
  do {                                                                    \
    static std::atomic<int> LOG_OCCURRENCES(0), LOG_OCCURRENCES_MOD_N(0); \
    ++LOG_OCCURRENCES;                                                    \
    if (++LOG_OCCURRENCES_MOD_N > n) LOG_OCCURRENCES_MOD_N -= n;          \
    if (LOG_OCCURRENCES_MOD_N == 1) {                                     \
      RUDF_LOG(level, __VA_ARGS__);                                       \
    }                                                                     \
  } while (0)

#define RUDF_INFO_EVERY_N(N, ...) RUDF_LOG_EVERY_N(spdlog::level::info, N, __VA_ARGS__)
#define RUDF_ERROR_EVERY_N(N, ...) RUDF_LOG_EVERY_N(spdlog::level::err, N, __VA_ARGS__)

#define RUDF_LOG_ERROR_STATUS(status)                 \
  do {                                                \
    auto return_status = (status);                    \
    RUDF_ERROR("Error:{}", return_status.ToString()); \
    return return_status;                             \
  } while (0)

#define RUDF_LOG_RETURN_ERROR_STATUS(status)            \
  do {                                                  \
    auto return_status = (status);                      \
    if (!return_status.ok()) {                          \
      RUDF_ERROR("Error:{}", return_status.ToString()); \
      return status;                                    \
    }                                                   \
  } while (0)

#define RUDF_LOG_RETURN_FMT_ERROR(...)                           \
  do {                                                           \
    RUDF_ERROR(__VA_ARGS__);                                     \
    return absl::InvalidArgumentError(fmt::format(__VA_ARGS__)); \
  } while (0)

#define RUDF_RETURN_FMT_ERROR(...)                               \
  do {                                                           \
    return absl::InvalidArgumentError(fmt::format(__VA_ARGS__)); \
  } while (0)
