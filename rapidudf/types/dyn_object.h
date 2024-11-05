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
  absl::StatusOr<T> Get(const std::string& name, uint32_t* offset = nullptr) const;

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