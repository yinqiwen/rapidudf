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
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/compiler/global_compiler.h"
#include "rapidudf/compiler/options.h"
#include "rapidudf/context/context.h"
#include "rapidudf/log/log.h"
#include "rapidudf/reflect/macros.h"
#include "rapidudf/types/dyn_object_impl.h"
#include "rapidudf/vector/table.h"
#include "rapidudf/vector/table_schema.h"
#include "rapidudf/version.h"

namespace rapidudf {
using JitCompiler = compiler::JitCompiler;
// using JitCompilerCache = llvm::JitCompilerCache;

template <typename RET, typename... Args>
using JitFunction = compiler::JitFunction<RET, Args...>;

using Options = compiler::Options;

using GlobalJitCompiler = compiler::GlobalJitCompiler;

}  // namespace rapidudf