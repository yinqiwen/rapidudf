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