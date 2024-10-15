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
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
namespace rapidudf {

template <typename T>
absl::Status DynObject::DoSet(const std::string& name, const T& v) {
  auto result = schema_->GetField(name);
  if (!result.ok()) {
    return result.status();
  }
  auto [dtype, offset] = result.value();
  if (get_dtype<T>() != dtype) {
    RUDF_RETURN_FMT_ERROR("[DynObject::Set]mismatch dtype:{} for field:{} with dtype:{}", get_dtype<T>(), name, dtype);
  }
  uint8_t* p = reinterpret_cast<uint8_t*>(this) + offset;
  *(reinterpret_cast<T*>(p)) = v;
  return absl::OkStatus();
}
template <typename T>
absl::Status DynObject::DoSet(const std::string& name, T&& v) {
  auto result = schema_->GetField(name);
  if (!result.ok()) {
    return result.status();
  }
  auto [dtype, offset] = result.value();
  if (get_dtype<T>() != dtype) {
    RUDF_RETURN_FMT_ERROR("[DynObject::Set]mismatch dtype:{} for field:{} with dtype:{}", get_dtype<T>(), name, dtype);
  }
  uint8_t* p = reinterpret_cast<uint8_t*>(this) + offset;
  *(reinterpret_cast<T*>(p)) = v;
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<T> DynObject::Get(const std::string& name) const {
  auto result = schema_->GetField(name);
  if (!result.ok()) {
    return result.status();
  }
  auto [dtype, offset] = result.value();
  if (get_dtype<T>() != dtype) {
    RUDF_RETURN_FMT_ERROR("[DynObject::Get]mismatch dtype:{} for field:{} with dtype:{}", get_dtype<T>(), name, dtype);
  }
  const uint8_t* p = reinterpret_cast<const uint8_t*>(this) + offset;
  return *(reinterpret_cast<const T*>(p));
}
}  // namespace rapidudf