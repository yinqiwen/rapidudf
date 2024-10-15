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
#include <chrono>
#include <memory>
#include <string>

namespace rapidudf {
namespace compiler {
struct JitFunctionStat {
  std::chrono::microseconds parse_cost;
  std::chrono::microseconds parse_validate_cost;
  std::chrono::microseconds ir_build_cost;
  std::chrono::microseconds compile_cost;
  void Clear() {
    parse_cost = std::chrono::microseconds::zero();
    parse_validate_cost = std::chrono::microseconds::zero();
    ir_build_cost = std::chrono::microseconds::zero();
    compile_cost = std::chrono::microseconds::zero();
  }
};

template <typename RET, typename... Args>
class JitFunction {
 public:
  JitFunction() = default;
  template <typename T>
  explicit JitFunction(const std::string& name, const void* f, std::shared_ptr<T> resource, const JitFunctionStat& stat,
                       bool from_cache = false)
      : name_(name), resource_(resource), stat_(stat), is_from_cache_(from_cache) {
    f_ = reinterpret_cast<RET (*)(Args...)>(const_cast<void*>(f));
  }
  JitFunction(JitFunction&& other) { MoveFrom(std::move(other)); }
  ~JitFunction() {}
  JitFunction(const JitFunction&) = delete;
  JitFunction& operator=(const JitFunction&) = delete;
  JitFunction& operator=(JitFunction&& other) {
    MoveFrom(std::move(other));
    return *this;
  }
  const JitFunctionStat& Stats() const { return stat_; }
  const std::string& GetName() const { return name_; }
  bool IsFromCache() const { return is_from_cache_; }

  RET operator()(Args... args) {
    if constexpr (std::is_same_v<void, RET>) {
      f_(args...);
    } else {
      RET r = f_(args...);
      return r;
    }
  }

 private:
  std::string name_;
  std::shared_ptr<void> resource_;
  RET (*f_)(Args...) = nullptr;
  JitFunctionStat stat_;
  bool is_from_cache_;

  void MoveFrom(JitFunction&& other) {
    name_ = std::move(other.name_);
    resource_ = std::move(other.resource_);
    f_ = other.f_;
    stat_ = other.stat_;
  }
};
}  // namespace compiler
}  // namespace rapidudf