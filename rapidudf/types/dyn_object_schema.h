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
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/types/dyn_object.h"

namespace rapidudf {
class DynObjectSchema {
 public:
  using InitFunc = std::function<void(DynObjectSchema* s)>;
  static const DynObjectSchema* GetOrCreate(const std::string& name, InitFunc&& init, size_t reserved_size = 0);
  static const DynObjectSchema* Get(const std::string& name);
  static DynObjectSchema* GetMutable(const std::string& name);
  static std::vector<std::string> ListAll();

  typename DynObject::SmartPtr NewObject() const;

  template <typename T>
  absl::Status AddField(const std::string& name) {
    return Add(name, get_dtype<T>());
  }

  absl::StatusOr<std::pair<DType, uint32_t>> GetField(const std::string& name) const;

  uint32_t ByteSize() const { return allocated_offset_; }

  bool IsTable() const { return flags_.is_table; }

  void VisitField(std::function<void(const std::string&, const DType&, uint32_t)>&& f) const;

  size_t FieldCount() const { return fields_.size(); }

 protected:
  struct Flags {
    uint64_t is_table : 1;
    uint64_t reserved : 63;
    Flags(bool table = false) { is_table = table ? 1 : 0; }
  };
  DynObjectSchema(const std::string& name, size_t reserved_size, Flags flags);
  void SetFlags(Flags flags) { flags_ = flags; }

  absl::Status Add(const std::string& name, DType dtype);

  struct Field {
    DType dtype;
    uint32_t bytes_offset = 0;
  };

  using FieldTable = absl::flat_hash_map<std::string, Field>;
  std::string name_;
  Flags flags_;
  FieldTable fields_;
  uint32_t allocated_offset_ = 0;
};
using DynObjectSchemaMap = std::map<std::string, DynObjectSchema*>;
}  // namespace rapidudf