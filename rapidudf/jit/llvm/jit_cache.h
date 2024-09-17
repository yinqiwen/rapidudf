/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once

#include "absl/container/flat_hash_map.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/log/log.h"

namespace rapidudf {
namespace llvm {
struct JitSession;
class JitCompilerCache {
 public:
  template <typename RET, typename... Args>
  static absl::StatusOr<JitFunction<RET, Args...>> GetExpression(const std::string& source,
                                                                 const std::vector<std::string>& arg_names) {
    auto& cache_map = GetCache();
    auto& cache_mutex = GetCacheMutex();
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    {
      std::lock_guard<std::mutex> guard(cache_mutex);
      auto found = cache_map.find(source);
      if (found != cache_map.end()) {
        auto& cache_item = found->second;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        for (auto& cache_func : cache_item.funcs) {
          if (cache_func.desc.CompareSignature(return_type, arg_types)) {
            RUDF_DEBUG("Cache hit for key:{}", source);
            return JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_item.session, false);
          }
        }
      }
    }
    JitCompiler compiler;
    auto result = compiler.CompileExpression<RET, Args...>(source, arg_names);
    if (!result.ok()) {
      return result.status();
    }
    auto ret_func = std::move(result.value());
    CacheFunction cache_func;
    cache_func.expr_session = compiler.session_;
    auto func_ptr_result = compiler.GetFunctionPtr(std::string(JitCompiler::kExpressionFuncName));
    if (!func_ptr_result.ok()) {
      return func_ptr_result.status();
    }
    cache_func.func = func_ptr_result.value();
    cache_func.desc.name = std::string(JitCompiler::kExpressionFuncName);
    cache_func.desc.arg_types = arg_types;
    cache_func.desc.return_type = return_type;
    {
      std::lock_guard<std::mutex> guard(cache_mutex);
      auto found = cache_map.find(source);
      if (found != cache_map.end()) {
        auto& cache_item = found->second;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        cache_item.funcs.emplace_back(cache_func);
      } else {
        CacheItem cache_item;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        cache_item.funcs.emplace_back(cache_func);
        cache_map[source] = cache_item;
      }
    }
    return ret_func;
  }
  template <typename RET, typename... Args>
  static absl::StatusOr<JitFunction<RET, Args...>> GetFunction(const std::string& source) {
    auto& cache_map = GetCache();
    auto& cache_mutex = GetCacheMutex();
    auto return_type = get_dtype<RET>();
    std::vector<DType> arg_types;
    (arg_types.emplace_back(get_dtype<Args>()), ...);
    {
      std::lock_guard<std::mutex> guard(cache_mutex);
      auto found = cache_map.find(source);
      if (found != cache_map.end()) {
        auto& cache_item = found->second;
        cache_item.latest_visit_time = std::chrono::high_resolution_clock::now();
        for (auto& cache_func : cache_item.funcs) {
          if (cache_func.desc.CompareSignature(return_type, arg_types)) {
            RUDF_DEBUG("Cache hit for key:{}", source);
            return JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_item.session, false);
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
    cache_item.session = compiler.session_;

    auto all_funcs = compiler.GetAllFunctionDescs();
    std::optional<JitFunction<RET, Args...>> found_func;
    for (auto& desc : all_funcs) {
      auto func_ptr_result = compiler.GetFunctionPtr(desc.name);
      if (!func_ptr_result.ok()) {
        return func_ptr_result.status();
      }
      auto func_ptr = func_ptr_result.value();
      CacheFunction cache_func;
      cache_func.desc = desc;
      cache_func.func = func_ptr;
      cache_item.funcs.emplace_back(cache_func);
      if (cache_func.desc.CompareSignature(return_type, arg_types)) {
        found_func = JitFunction<RET, Args...>(cache_func.desc.name, cache_func.func, cache_item.session, false);
      }
    }
    {
      std::lock_guard<std::mutex> guard(cache_mutex);
      cache_map[source] = cache_item;
    }
    if (found_func.has_value()) {
      return std::move(*found_func);
    }
    return absl::NotFoundError("No func found in compiled source.");
  }

  // slow function
  static size_t RemoveExpiredCache(std::chrono::seconds ttl_secs);

 private:
  struct CacheFunction {
    FunctionDesc desc;
    void* func = nullptr;
    std::shared_ptr<JitSession> expr_session;
  };
  struct CacheItem {
    std::shared_ptr<JitSession> session;
    std::vector<CacheFunction> funcs;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> latest_visit_time;
    void Merge(CacheItem& other) {}
  };
  using CacheMap = absl::flat_hash_map<std::string, CacheItem>;
  static CacheMap& GetCache() {
    static CacheMap cache;
    return cache;
  }
  static std::mutex& GetCacheMutex() {
    static std::mutex cache_mutex;
    return cache_mutex;
  }
};
}  // namespace llvm
}  // namespace rapidudf