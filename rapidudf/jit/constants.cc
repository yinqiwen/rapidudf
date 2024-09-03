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

#include <cstdint>
#include <memory>
#include "rapidudf/codegen/dtype.h"
#include "rapidudf/codegen/ops/cast.h"
#include "rapidudf/jit/jit.h"
#include "rapidudf/log/log.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
using namespace Xbyak::util;

absl::StatusOr<ValuePtr> JitCompiler::CompileConstants(double v, DType dtype) {
  auto val = GetCodeGenerator().NewConstValue(dtype);
  if (dtype.IsInteger()) {
    int64_t iv = static_cast<int64_t>(v);
    val->Set(iv);
  } else if (dtype.IsF32()) {
    val->Set(static_cast<float>(v));
  } else {
    val->Set(v);
  }
  return val;
}
absl::StatusOr<ValuePtr> JitCompiler::CompileConstants(double v) {
  int64_t iv = static_cast<int64_t>(v);
  if (static_cast<double>(iv) == v) {
    auto val = GetCodeGenerator().NewConstValue(DType(iv <= INT32_MAX ? DATA_I32 : DATA_I64));
    val->Set(iv);
    return val;
  }
  auto val = GetCodeGenerator().NewConstValue(DType(DATA_F64));
  val->Set(v);
  return val;
}

absl::StatusOr<ValuePtr> JitCompiler::CompileConstants(bool v) {
  auto val = GetCodeGenerator().NewConstValue(DType(DATA_U8));
  val->Set(v ? 1 : 0);
  return val;
}
absl::StatusOr<ValuePtr> JitCompiler::CompileConstants(const std::string& v) {
  std::unique_ptr<std::string> str = std::make_unique<std::string>(v);
  StringView view(*str);
  GetCompileContext().const_strings.emplace_back(std::move(str));
  auto val = GetCodeGenerator().NewConstValue(DType(DATA_STRING_VIEW));
  if (0 != val->Set(view)) {
    GetCodeGenerator().DropTmpValue(val);
    RUDF_LOG_ERROR_STATUS(
        ast_ctx_.GetErrorStatus(fmt::format("Can NOT set const string '{}' to val with dtype:{}", v, val->GetDType())));
  }
  return val;
}
}  // namespace rapidudf