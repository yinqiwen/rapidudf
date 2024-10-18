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
#include "rapidudf/ast/function.h"
#include <fmt/core.h>
#include "rapidudf/log/log.h"
#include "rapidudf/types/dyn_object.h"
namespace rapidudf {
namespace ast {
FunctionDesc Function::ToFuncDesc() const {
  FunctionDesc desc;
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
      if (!arg.attr.schema.empty()) {
        arg.schema = DynObjectSchema::Get(arg.attr.schema);
        if (arg.schema == nullptr) {
          RUDF_RETURN_FMT_ERROR("No dyn_obj schema:{} found for arg:{}", arg.attr.schema, arg.name);
        }
      }
      ctx.AddLocalVar(arg.name, arg.dtype, arg.schema);
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