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
#include <memory>
#include <string>
#include <string_view>

#include "rapidudf/common/lru_cache.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/compiler/function.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace exec {

struct EvalFunction {
  FunctionDesc desc;
  void* func_ptr = nullptr;
  compiler::JitFunctionStat stat;
  std::shared_ptr<compiler::CodeGen> codegen;
  std::shared_ptr<void> func_obj;
  template <typename FUNC>
  FUNC* GetFunc() {
    if (!func_obj) {
      func_obj = std::make_shared<FUNC>(desc.name, func_ptr, codegen, stat);
    }
    return reinterpret_cast<FUNC*>(func_obj.get());
  }
};
struct EvalCacheValue {
  std::vector<EvalFunction> funcs;
  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> latest_visit_time;
};
struct EvalCache : lru_cache<std::string, EvalCacheValue> {
  std::mutex mutex;
  explicit EvalCache(size_t n) : lru_cache<std::string, EvalCacheValue>(n) {}
};

EvalCache& get_eval_cache();
template <class R, class... Args>
absl::StatusOr<R> eval_function(const std::string& source, Args... args) {
  using FUNC = compiler::JitFunction<R, Args...>;
  FUNC* found_func = nullptr;
  auto& cache_map = get_eval_cache();
  auto return_type = get_dtype<R>();
  std::vector<DType> arg_types;
  (arg_types.emplace_back(get_dtype<Args>()), ...);

  {
    std::lock_guard<std::mutex> guard(cache_map.mutex);
    auto found = cache_map.get(source);
    if (found) {
      auto& cache_item = *found;
      cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
      for (auto& cache_func : cache_item.funcs) {
        if (cache_func.desc.CompareSignature(return_type, arg_types)) {
          RUDF_DEBUG("Cache hit for key:{}", source);
          found_func = cache_func.GetFunc<FUNC>();
          return (*found_func)(args...);
        }
      }
      return absl::NotFoundError("No func found in cache.");
    }
  }
  compiler::JitCompiler compiler;
  auto result = compiler.CompileSource(source);
  if (!result.ok()) {
    return result.status();
  }
  EvalCacheValue cache_item;
  cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
  auto all_funcs = compiler.GetParsedAST();
  for (auto& func : all_funcs) {
    auto func_ptr_result = compiler.GetFunctionPtr(func.name);
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto func_ptr = func_ptr_result.value();
    EvalFunction cache_func;
    cache_func.desc = func.ToFuncDesc();
    cache_func.func_ptr = func_ptr;
    cache_func.codegen = compiler.GetCodeGen();
    cache_func.stat = compiler.GetStat();
    cache_item.funcs.emplace_back(cache_func);
    if (cache_func.desc.CompareSignature(return_type, arg_types)) {
      found_func = cache_func.GetFunc<FUNC>();
    }
  }
  {
    std::lock_guard<std::mutex> guard(cache_map.mutex);
    cache_map.insert(source, cache_item);
  }
  if (found_func) {
    return (*found_func)(args...);
  }
  return absl::NotFoundError("No func found in compiled source.");
}

template <class R, class... Args>
absl::StatusOr<R> eval_expression(const std::string& source, const std::vector<compiler::JitCompiler::Arg>& arg_descs,
                                  Args... args) {
  using FUNC = compiler::JitFunction<R, Args...>;
  FUNC* found_func = nullptr;
  auto& cache_map = get_eval_cache();
  {
    std::lock_guard<std::mutex> guard(cache_map.mutex);
    auto found = cache_map.get(source);
    if (found) {
      auto return_type = get_dtype<R>();
      std::vector<DType> arg_types;
      (arg_types.emplace_back(get_dtype<Args>()), ...);
      auto& cache_item = *found;
      cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
      for (auto& cache_func : cache_item.funcs) {
        if (cache_func.desc.CompareSignature(return_type, arg_types)) {
          RUDF_DEBUG("Cache hit for key:{}", source);
          found_func = cache_func.GetFunc<FUNC>();
          return (*found_func)(args...);
        }
      }
    }
  }
  compiler::JitCompiler compiler;
  auto result = compiler.CompileDynObjExpression<R, Args...>(source, arg_descs);
  if (!result.ok()) {
    return result.status();
  }
  auto ret_func = std::move(result.value());
  EvalFunction cache_func;
  cache_func.codegen = compiler.GetCodeGen();
  auto func_ptr_result = compiler.GetFunctionPtr(std::string(compiler::JitCompiler::kExpressionFuncName));
  if (!func_ptr_result.ok()) {
    return func_ptr_result.status();
  }
  auto return_type = get_dtype<R>();
  std::vector<DType> arg_types;
  (arg_types.emplace_back(get_dtype<Args>()), ...);
  cache_func.func_ptr = func_ptr_result.value();
  cache_func.desc.name = std::string(compiler::JitCompiler::kExpressionFuncName);
  cache_func.desc.arg_types = arg_types;
  cache_func.desc.return_type = return_type;
  cache_func.stat = compiler.GetStat();
  found_func = cache_func.GetFunc<FUNC>();
  {
    std::lock_guard<std::mutex> guard(cache_map.mutex);
    auto found = cache_map.get(source);
    if (found) {
      auto& cache_item = *found;
      cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
      cache_item.funcs.emplace_back(cache_func);
    } else {
      EvalCacheValue cache_item;
      cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
      cache_item.funcs.emplace_back(cache_func);
      cache_map.insert(source, cache_item);
    }
  }
  return (*found_func)(args...);
}

template <class R, class... Args>
absl::StatusOr<R> eval_expression(const std::string& source, const std::vector<std::string>& arg_names, Args... args) {
  std::vector<compiler::JitCompiler::Arg> arg_descs;
  for (auto& name : arg_names) {
    arg_descs.emplace_back(compiler::JitCompiler::Arg{.name = name});
  }
  return eval_expression<R, Args...>(source, arg_descs, args...);
}

}  // namespace exec
}  // namespace rapidudf