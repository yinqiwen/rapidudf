/*
 *Copyright (c) 2021, qiyingwang <qiyingwang@tencent.com>
 *All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of rimos nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 *BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 *THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rapidudf/types/params.h"
#include <string_view>

namespace rapidudf {

Params::Params(bool invalid) : iv_(0), dv_(0), bv_(false), parent_(nullptr) {
  if (invalid) {
    json_type_ = JSON_INVALID;
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
bool Params::Valid() const { return json_type_ != JSON_INVALID; }
bool Params::IsBool() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_BOOL;
}
bool Params::IsString() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_STRING;
}
bool Params::IsDouble() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_DOUBLE;
}
bool Params::IsInt() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_INT;
}
bool Params::IsObject() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_OBJECT;
}
bool Params::IsArray() const {
  if (!Valid()) {
    return false;
  }
  return json_type_ == JSON_ARRAY;
}
const ParamString& Params::String() const { return str_; }
int64_t Params::Int() const { return iv_; }
bool Params::Bool() const { return bv_; }
double Params::Double() const { return dv_; }
void Params::SetString(const ParamString& v) {
  str_ = v;
  json_type_ = JSON_STRING;
}
void Params::SetInt(int64_t v) {
  iv_ = v;
  json_type_ = JSON_INT;
}
void Params::SetDouble(double d) {
  dv_ = d;
  json_type_ = JSON_DOUBLE;
}
void Params::SetBool(bool v) {
  bv_ = v;
  json_type_ = JSON_BOOL;
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
  json_type_ = JSON_OBJECT;
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
  json_type_ = JSON_ARRAY;
  return array_[array_.size() - 1];
}
Params& Params::Put(const ParamString& name, const char* value) {
  json_type_ = JSON_OBJECT;
  members_[name].SetString(value);
  return *this;
}
Params& Params::Put(const ParamString& name, const ParamString& value) {
  json_type_ = JSON_OBJECT;
  members_[name].SetString(value);
  return *this;
}
Params& Params::Put(const ParamString& name, int64_t value) {
  json_type_ = JSON_OBJECT;
  members_[name].SetInt(value);
  return *this;
}
Params& Params::Put(const ParamString& name, double value) {
  json_type_ = JSON_OBJECT;
  members_[name].SetDouble(value);
  return *this;
}
Params& Params::Put(const ParamString& name, bool value) {
  json_type_ = JSON_OBJECT;
  members_[name].SetBool(value);
  return *this;
}
Params& Params::operator[](size_t idx) {
  if (IsArray()) {
    if (array_.size() > idx) {
      return array_[idx];
    }
  }
  json_type_ = JSON_ARRAY;
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