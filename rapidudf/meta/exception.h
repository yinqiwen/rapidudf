/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <stdexcept>
#include <string>
#include "fmt/format.h"

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
