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
#include <memory>
#include <string>
#include <utility>
#include "absl/status/statusor.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
class DynObjectSchema;
class DynObject {
 private:
  struct Deleter {
    void operator()(DynObject* ptr) {
      ptr->~DynObject();
      uint8_t* bytes = reinterpret_cast<uint8_t*>(ptr);
      delete[] bytes;
    }
  };

 public:
  typedef std::unique_ptr<DynObject, Deleter> SmartPtr;
  template <typename T>
  absl::Status Set(const std::string& name, T&& v) {
    return DoSet(name, std::forward<T>(v));
  }
  absl::Status Set(const std::string& name, const char* sv) { return DoSet(name, StringView(sv)); }

  template <typename T>
  absl::StatusOr<T> Get(const std::string& name) const;

 protected:
  DynObject(const DynObjectSchema* s) : schema_(s) {}
  template <typename T>
  absl::Status DoSet(const std::string& name, const T& v);
  template <typename T>
  absl::Status DoSet(const std::string& name, T&& v);

  const DynObjectSchema* schema_ = nullptr;
  friend class DynObjectSchema;
};
}  // namespace rapidudf