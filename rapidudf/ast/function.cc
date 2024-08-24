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
#include "rapidudf/ast/function.h"
#include <fmt/core.h>
#include "rapidudf/log/log.h"
namespace rapidudf {
namespace ast {
FuncDesc Function::ToFuncDesc() const {
  FuncDesc desc;
  desc.name = name;
  desc.return_type = return_type;
  if (args.has_value()) {
    for (auto& arg : *args) {
      desc.arg_types.emplace_back(arg.dtype);
    }
  }
  return desc;
}
absl::Status Function::Validate(ParseContext& ctx) {
  ctx.SetPosition(position);

  if (args.has_value()) {
    for (auto& arg : *args) {
      auto result = ctx.IsVarExist(arg.name, true);
      if (!result.ok()) {
        return result.status();
      }
      ctx.AddLocalVar(arg.name, arg.dtype);
    }
  }
  return body.Validate(ctx);
}
bool Function::CompareSignature(DType rtype, const std::vector<DType>& args_types, std::string& err) {
  if (return_type != rtype) {
    err = fmt::format("function:{} mismatch signature on return data type, while expect {}, but given {}", name,
                      return_type, rtype);
    return false;
  }
  if (!args.has_value()) {
    if (args_types.empty()) {
      return true;
    }
    err = fmt::format("function:{} mismatch signature on args size, while expect {}, but given {}", name, 0,
                      args_types.size());
    return false;
  } else {
    if (args_types.size() != args->size()) {
      err = fmt::format("function:{} mismatch signature on args size, while expect {}, but given {}", name,
                        args->size(), args_types.size());
      return false;
    }
    for (size_t i = 0; i < args_types.size(); i++) {
      if (args_types[i] != args->at(i).dtype) {
        err = fmt::format("function:{} mismatch signature on args[{}] data type, while expect {}, but given {}", name,
                          i, args->at(i).dtype, args_types[i]);
        return false;
      }
    }
  }
  return true;
}
}  // namespace ast

}  // namespace rapidudf