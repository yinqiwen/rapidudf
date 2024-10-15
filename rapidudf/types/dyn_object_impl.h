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