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
#include "rapidudf/reflect/struct_access.h"
#include <fmt/core.h>
#include <functional>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/function.h"
#include "rapidudf/codegen/value.h"
#include "rapidudf/log/log.h"
#include "xbyak/xbyak_util.h"
namespace rapidudf {
using namespace Xbyak::util;
// StructFieldReflectAccessJitBuilder get_field_read_code_builder(DType member_dtype, uint32_t offset) {
//   StructFieldReflectAccessJitBuilder f;
//   f = [=](Xbyak::CodeGenerator& c, DType& dtype) -> std::vector<const Xbyak::Reg*> {
//     // rdi store struct ptr
//     uint32_t bits = member_dtype.Bits();
//     dtype = member_dtype;
//     RUDF_DEBUG("access field dtype:{}", member_dtype);
//     c.add(rcx, offset);
//     if (member_dtype.IsNumber()) {
//       // return number value
//       c.mov(rax.changeBit(bits), ptr[rcx]);
//       return {&rax};
//     } else if (member_dtype.IsPtr()) {
//       // return ptr value
//       c.mov(rax, ptr[rcx]);
//       return {&rax};
//     } else if (member_dtype.IsAbslSpan()) {
//       c.mov(rax, ptr[rcx]);
//       c.add(rcx, 8);
//       c.mov(rdx, ptr[rcx]);
//       return {&rax, &rdx};
//     } else if (member_dtype == DATA_STRING_VIEW) {
//       c.mov(rax, ptr[rcx]);
//       c.add(rcx, 8);
//       c.mov(rdx, ptr[rcx]);
//       return {&rax, &rdx};
//       // typedef void (*string_view_data_access)(std::string_view*);
//       // typedef void (*string_view_size_access)(std::string_view*);
//       // string_view_data_access data_ptr = reinterpret_cast<string_view_data_access>(&std::string_view::data);
//       // string_view_data_access size_ptr = reinterpret_cast<string_view_data_access>(&std::string_view::size);
//       // c.mov(rax, (size_t)data_ptr);
//       // c.call(rax);
//       // c.mov(rdx, rax);
//       // c.mov(rax, (size_t)size_ptr);
//       // c.call(rax);
//     } else {
//       // return ptr for normal object
//       c.mov(rax, rcx);
//       dtype.ToPtr();
//       return {&rax};
//     }
//   };
//   return f;
// }
// StructFieldReflectAccessJitBuilder get_field_write_code_builder(DType member_dtype, uint32_t offset) {
//   StructFieldReflectAccessJitBuilder f;
//   f = [=](Xbyak::CodeGenerator& c, DType& dtype) -> std::vector<const Xbyak::Reg*> {
//     dtype = member_dtype;
//     RUDF_DEBUG("access field dtype:{}", member_dtype);
//     c.add(rcx, offset);
//     c.mov(rax, rcx);
//     dtype.ToPtr();
//     return {&rax};
//   };
//   return f;
// }

}  // namespace rapidudf