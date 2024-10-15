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

#pragma once

#include <stdexcept>
#include <string>
#include "fmt/format.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

class UDFRuntimeException : public std::logic_error {
 public:
  explicit UDFRuntimeException(const std::string& msg) : std::logic_error(msg) {}
};

class NullPointerException : public UDFRuntimeException {
 public:
  explicit NullPointerException(const std::string& msg) : UDFRuntimeException(msg) {}
};
class ReadonlyException : public UDFRuntimeException {
 public:
  explicit ReadonlyException(const std::string& msg) : UDFRuntimeException(msg) {}
};
class SizeMismatchException : public UDFRuntimeException {
 public:
  explicit SizeMismatchException(size_t current, size_t expect, const std::string& msg)
      : UDFRuntimeException(fmt::format("expect size:{}, but got size:{} at {}", expect, current, msg)) {}
};

class OutOfRangeException : public UDFRuntimeException {
 public:
  explicit OutOfRangeException(size_t requested, size_t limit, const std::string& msg)
      : UDFRuntimeException(fmt::format("limit size:{}, but requested:{} at {}", limit, requested, msg)) {}
};
class VectorExpressionException : public UDFRuntimeException {
 public:
  explicit VectorExpressionException(int line, StringView src_line, StringView msg)
      : UDFRuntimeException(fmt::format("Error:{} occurred at line:{}, source ' {} '", msg, line, src_line)) {}
};
}  // namespace rapidudf

#define THROW_LOGIC_ERR(msg)                                                        \
  do {                                                                              \
    throw std::logic_error(fmt::format("{}:{} error:{}", __FILE__, __LINE__, msg)); \
  } while (0)

#define THROW_NULL_POINTER_ERR(msg)                                                               \
  do {                                                                                            \
    throw rapidudf::NullPointerException(fmt::format("{}:{} error:{}", __FILE__, __LINE__, msg)); \
  } while (0)

#define THROW_READONLY_ERR(msg)                                                                \
  do {                                                                                         \
    throw rapidudf::ReadonlyException(fmt::format("{}:{} error:{}", __FILE__, __LINE__, msg)); \
  } while (0)

#define THROW_SIZE_MISMATCH_ERR(current, expect)                                                      \
  do {                                                                                                \
    throw rapidudf::SizeMismatchException(current, expect, fmt::format("{}:{}", __FILE__, __LINE__)); \
  } while (0)

#define THROW_OUT_OF_RANGE_ERR(requested, limit)                                                     \
  do {                                                                                               \
    throw rapidudf::OutOfRangeException(requested, limit, fmt::format("{}:{}", __FILE__, __LINE__)); \
  } while (0)
