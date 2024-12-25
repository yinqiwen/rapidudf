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

#include "rapidudf/types/params.h"
#include <string_view>

namespace rapidudf {

Params::Params(bool invalid) : iv_(0), dv_(0), bv_(false), parent_(nullptr) {
  if (invalid) {
    param_type_ = PARAM_INVALID;
  }
}
void Params::SetParent(const Params* p) {
  if (nullptr == p) {
    parent_ = nullptr;
    return;
  }
  if (this == p) {
    return;
  }
  if (nullptr == parent_) {
    parent_ = p;
  } else {
    if (parent_ == p) {
      return;
    }
    const_cast<Params*>(parent_)->SetParent(p);
  }
}
bool Params::Valid() const { return param_type_ != PARAM_INVALID; }
bool Params::IsBool() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_BOOL;
}
bool Params::IsString() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_STRING;
}
bool Params::IsDouble() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_DOUBLE;
}
bool Params::IsInt() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_INT;
}
bool Params::IsObject() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_OBJECT;
}
bool Params::IsArray() const {
  if (!Valid()) {
    return false;
  }
  return param_type_ == PARAM_ARRAY;
}
const ParamString& Params::String() const { return str_; }
int64_t Params::Int() const { return iv_; }
bool Params::Bool() const { return bv_; }
double Params::Double() const { return dv_; }
void Params::SetString(const ParamString& v) {
  str_ = v;
  param_type_ = PARAM_STRING;
}
void Params::SetInt(int64_t v) {
  iv_ = v;
  param_type_ = PARAM_INT;
}
void Params::SetDouble(double d) {
  dv_ = d;
  param_type_ = PARAM_DOUBLE;
}
void Params::SetBool(bool v) {
  bv_ = v;
  param_type_ = PARAM_BOOL;
}
size_t Params::Size() const {
  if (IsObject()) {
    return members_.size();
  } else if (IsArray()) {
    return array_.size();
  }
  return 0;
}
const typename Params::ParamsTable& Params::Members() const { return members_; }
const Params& Params::Get(std::string_view name) const {
  auto it = members_.find(name);
  if (it != members_.end()) {
    return it->second;
  }
  if (nullptr != parent_) {
    return parent_->Get(name);
  }
  static Params default_value(true);
  return default_value;
}
const Params& Params::Get(const ParamString& name) const {
  std::string_view name_view = name;
  return Get(name_view);
}
const Params& Params::operator[](std::string_view name) const { return Get(name); }
const Params& Params::operator[](const ParamString& name) const { return Get(name); }
Params& Params::operator[](const ParamString& name) {
  const Params& p = Get(name);
  if (p.Valid()) {
    return const_cast<Params&>(p);
  }
  param_type_ = PARAM_OBJECT;
  return members_[name];
}
const Params& Params::operator[](size_t idx) const {
  if (IsArray()) {
    if (array_.size() > idx) {
      return array_[idx];
    }
  }
  static Params default_value(true);
  return default_value;
}
Params& Params::Add() {
  array_.resize(array_.size() + 1);
  param_type_ = PARAM_ARRAY;
  return array_[array_.size() - 1];
}
Params& Params::Put(const ParamString& name, const char* value) {
  param_type_ = PARAM_OBJECT;
  members_[name].SetString(value);
  return *this;
}
Params& Params::Put(const ParamString& name, const ParamString& value) {
  param_type_ = PARAM_OBJECT;
  members_[name].SetString(value);
  return *this;
}
Params& Params::Put(const ParamString& name, int64_t value) {
  param_type_ = PARAM_OBJECT;
  members_[name].SetInt(value);
  return *this;
}
Params& Params::Put(const ParamString& name, double value) {
  param_type_ = PARAM_OBJECT;
  members_[name].SetDouble(value);
  return *this;
}
Params& Params::Put(const ParamString& name, bool value) {
  param_type_ = PARAM_OBJECT;
  members_[name].SetBool(value);
  return *this;
}
Params& Params::operator[](size_t idx) {
  if (IsArray()) {
    if (array_.size() > idx) {
      return array_[idx];
    }
  }
  param_type_ = PARAM_ARRAY;
  array_.resize(idx + 1);
  return array_[idx];
}
bool Params::Contains(const ParamString& name) const {
  if (IsObject()) {
    if (members_.count(name) > 0) {
      return true;
    }
  }
  if (nullptr != parent_) {
    return parent_->Contains(name);
  }
  return false;
}

}  // namespace rapidudf