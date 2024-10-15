/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include "rapidudf/ast/statement.h"
namespace rapidudf {
namespace ast {
struct Block {
  std::vector<Statement> statements;
  absl::Status Validate(ParseContext& ctx) {
    for (auto& statement : statements) {
      auto status = std::visit([&](auto&& arg) { return arg.Validate(ctx); }, statement);
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace ast
}  // namespace rapidudf