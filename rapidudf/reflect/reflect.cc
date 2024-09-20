/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/reflect/reflect.h"
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {

static GlobalStructMemberIndex& get_global_reflect_index() {
  static GlobalStructMemberIndex index;
  return index;
}

std::optional<StructMember> Reflect::GetStructMember(DType dtype, const std::string& name) {
  auto found = get_global_reflect_index().find(dtype.Control());
  if (found == get_global_reflect_index().end()) {
    return {};
  }
  auto iter = found->second.find(name);
  if (iter == found->second.end()) {
    return {};
  }
  return iter->second;
}
bool Reflect::AddStructField(DType obj_dtype, const std::string& name, DType field_dtype, uint32_t field_offset) {
  auto [iter, success] =
      get_global_reflect_index()[obj_dtype.Control()].emplace(name, StructMember(name, field_dtype, field_offset));
  if (!success) {
    StructMember& entry = iter->second;
    if (!entry.member_field_dtype.has_value()) {
      entry.member_field_dtype = field_dtype;
      entry.member_field_offset = field_offset;
      entry.field_name = name;
      return true;
    }
  }
  return success;
}
bool Reflect::AddStructMethodAccessor(DType dtype, const std::string& name, const FunctionDesc& f) {
  auto [iter, success] = get_global_reflect_index()[dtype.Control()].emplace(name, StructMember(f));
  if (!success) {
    StructMember& entry = iter->second;
    if (!entry.member_func.has_value()) {
      entry.member_func = f;
      return true;
    }
  }
  return success;
}

}  // namespace rapidudf