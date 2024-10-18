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
#include <optional>
#include <string>
#include <vector>
#include "rapidudf/ast/block.h"
#include "rapidudf/ast/context.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/dyn_object_schema.h"

namespace rapidudf {
namespace ast {

struct FunctionArg {
  DType dtype;
  DTypeAttr attr;
  std::string name;
  const DynObjectSchema* schema = nullptr;
};
struct Function {
  DType return_type;
  std::string name;
  std::optional<std::vector<FunctionArg>> args;
  Block body;
  uint32_t position = 0;
  absl::Status Validate(ParseContext& ctx);
  bool CompareSignature(DType rtype, const std::vector<DType>& args_types, std::string& err);
  FunctionDesc ToFuncDesc() const;
};
}  // namespace ast
}  // namespace rapidudf