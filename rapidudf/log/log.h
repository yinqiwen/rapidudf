/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective
*contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <atomic>
#include <memory>
#include <string>

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
