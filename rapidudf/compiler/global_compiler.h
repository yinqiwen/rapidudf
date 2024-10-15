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

#include "rapidudf/common/lru_cache.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/compiler/function.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace compiler {

class GlobalJitCompiler {
 public:
  template <typename RET, typename... Args>
  static absl::StatusOr<JitFunction<RET, Args...>> GetExpression(const std::string& source,
                                                                 const std::vector<std::string>& arg_names) {
    std::vector<JitCompiler::Arg> args;
    for (auto& name : arg_names) {
      args.emplace_back(JitCompiler::Arg{.name = name});
    }
    return DoGetExpression<RET, Args...>(source, args);
  }
  template <typename RET, typename... Args>
  absl::StatusOr<JitFunction<RET, Args...>> GetDynObjExpression(const std::string& source,
                                                                const std::vector<JitCompiler::Arg>& args) {
    return DoGetExpression<RET, Args...>(source, args);
  }
  template <typename RET, typename... Args>
  static absl::StatusOr<JitFunction<RET, Args...>> GetFunction(const std::string& source) {
    auto& cache_map = GetCache();
    auto return_type = get_dtype<RET>();
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
            return JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_func.codegen, cache_func.stat,
                                             true);
          }
        }
        return absl::NotFoundError("No func found in cache.");
      }
    }
    JitCompiler compiler;
    auto result = compiler.CompileSource(source);
    if (!result.ok()) {
      return result.status();
    }
    CacheItem cache_item;
    cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();

    auto all_funcs = compiler.parsed_ast_funcs_;
    std::optional<JitFunction<RET, Args...>> found_func;
    for (auto& func : all_funcs) {
      auto func_ptr_result = compiler.GetFunctionPtr(func.name);
      if (!func_ptr_result.ok()) {
        return func_ptr_result.status();
      }
      auto func_ptr = func_ptr_result.value();
      CacheFunction cache_func;
      cache_func.desc = func.ToFuncDesc();
      cache_func.func = func_ptr;
      cache_func.codegen = compiler.codegen_;
      cache_func.stat = compiler.stat_;
      cache_item.funcs.emplace_back(cache_func);
      if (cache_func.desc.CompareSignature(return_type, arg_types)) {
        found_func =
            JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_func.codegen, cache_func.stat);
      }
    }
    {
      std::lock_guard<std::mutex> guard(cache_map.mutex);
      cache_map.insert(source, cache_item);
    }
    if (found_func.has_value()) {
      return std::move(*found_func);
    }
    return absl::NotFoundError("No func found in compiled source.");
  }

  static void ResetLRUCacheSize(size_t n) { GetCache().reset_capacity(n); }
  static size_t GetLRUCacheCapacity() { return GetCache().capacity(); }
  static size_t GetLRUCacheSize() { return GetCache().size(); }

 private:
  static const int kDefaultLRUCacheSize = 10000;

  struct CacheFunction {
    FunctionDesc desc;
    void* func = nullptr;
    JitFunctionStat stat;
    std::shared_ptr<CodeGen> codegen;
  };
  struct CacheItem {
    std::vector<CacheFunction> funcs;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> latest_visit_time;
    void Merge(CacheItem& other) {}
  };
  struct CacheMap : lru_cache<std::string, CacheItem> {
    std::mutex mutex;
    CacheMap(size_t n) : lru_cache<std::string, CacheItem>(n) {}
  };
  static CacheMap& GetCache() {
    static CacheMap cache(kDefaultLRUCacheSize);
    return cache;
  }

  template <typename RET, typename... Args>
  static absl::StatusOr<JitFunction<RET, Args...>> DoGetExpression(const std::string& source,
                                                                   const std::vector<JitCompiler::Arg>& args) {
    auto& cache_map = GetCache();

    {
      std::lock_guard<std::mutex> guard(cache_map.mutex);
      auto found = cache_map.get(source);
      if (found) {
        auto return_type = get_dtype<RET>();
        std::vector<DType> arg_types;
        (arg_types.emplace_back(get_dtype<Args>()), ...);
        auto& cache_item = *found;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        for (auto& cache_func : cache_item.funcs) {
          if (cache_func.desc.CompareSignature(return_type, arg_types)) {
            RUDF_DEBUG("Cache hit for key:{}", source);
            return JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_func.codegen, cache_func.stat,
                                             true);
          }
        }
      }
    }
    JitCompiler compiler;
    auto result = compiler.CompileDynObjExpression<RET, Args...>(source, args);
    if (!result.ok()) {
      return result.status();
    }
    auto ret_func = std::move(result.value());
    CacheFunction cache_func;
    cache_func.codegen = compiler.codegen_;
    auto func_ptr_result = compiler.GetFunctionPtr(std::string(JitCompiler::kExpressionFuncName));
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);

    cache_func.func = func_ptr_result.value();
    cache_func.desc.name = std::string(JitCompiler::kExpressionFuncName);
    cache_func.desc.arg_types = arg_types;
    cache_func.desc.return_type = return_type;
    cache_func.stat = compiler.stat_;
    {
      std::lock_guard<std::mutex> guard(cache_map.mutex);
      auto found = cache_map.get(source);
      if (found) {
        auto& cache_item = *found;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        cache_item.funcs.emplace_back(cache_func);
      } else {
        CacheItem cache_item;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        cache_item.funcs.emplace_back(cache_func);
        cache_map.insert(source, cache_item);
      }
    }
    return ret_func;
  }
};
}  // namespace compiler
}  // namespace rapidudf