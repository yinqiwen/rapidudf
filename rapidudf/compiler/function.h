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