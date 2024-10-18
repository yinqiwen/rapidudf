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
#include <stdint.h>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace rapidudf {
using ParamString = std::string;
class Params;
using ParamsPtr = std::shared_ptr<Params>;
class Params {
 public:
  using ParamsTable = absl::flat_hash_map<ParamString, Params>;
  using ParamsArray = std::vector<Params>;

 protected:
  enum ParamType {
    JSON_INVALID = 0,
    JSON_EMPTY,
    JSON_STRING,
    JSON_INT,
    JSON_DOUBLE,
    JSON_BOOL,
    JSON_OBJECT,
    JSON_ARRAY,
  };
  ParamType json_type_ = JSON_EMPTY;
  ParamString str_;
  int64_t iv_;
  double dv_;
  bool bv_;

  ParamsTable members_;
  ParamsArray array_;

  const Params* parent_ = nullptr;

 public:
  explicit Params(bool invalid_ = false);
  void SetParent(const Params* p);
  bool Valid() const;
  bool IsBool() const;
  bool IsString() const;
  bool IsDouble() const;
  bool IsInt() const;
  bool IsObject() const;
  bool IsArray() const;
  const ParamString& String() const;
  int64_t Int() const;
  bool Bool() const;
  double Double() const;
  void SetString(const ParamString& v);
  void SetInt(int64_t v);
  void SetDouble(double d);
  void SetBool(bool v);
  size_t Size() const;
  const ParamsTable& Members() const;
  uint32_t Type() const { return json_type_; }

  const Params& operator[](const ParamString& name) const;
  const Params& operator[](std::string_view name) const;
  Params& operator[](const ParamString& name);
  const Params& operator[](size_t idx) const;
  Params& operator[](size_t idx);

  Params& Add();
  const Params& Get(const ParamString& name) const;
  const Params& Get(std::string_view name) const;
  Params& Put(const ParamString& name, const char* value);
  Params& Put(const ParamString& name, const ParamString& value);
  Params& Put(const ParamString& name, int64_t value);
  Params& Put(const ParamString& name, double value);
  Params& Put(const ParamString& name, bool value);
  bool Contains(const ParamString& name) const;
};

}  // namespace rapidudf