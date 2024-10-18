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
#include <string>
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/function.h"
namespace rapidudf {
namespace ast {

absl::StatusOr<std::vector<Function>> parse_functions_ast(ParseContext& ctx, const std::string& source);
absl::StatusOr<Function> parse_function_ast(ParseContext& ctx, const std::string& source);
absl::StatusOr<Expression> parse_expression_ast(ParseContext& ctx, const std::string& source, const FunctionDesc& desc);

}  // namespace ast
}  // namespace rapidudf