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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"

namespace rapidudf {
namespace compiler {

::llvm::Type* get_type(::llvm::LLVMContext& ctx, DType dtype);
::llvm::VectorType* get_vector_type(::llvm::LLVMContext& ctx, DType dtype);

void init_buitin_types(::llvm::LLVMContext& ctx);

}  // namespace compiler
}  // namespace rapidudf