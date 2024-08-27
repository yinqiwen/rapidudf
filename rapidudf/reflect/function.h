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
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace rapidudf {

// static void func(std::string&& s) {}

// void test() {
//   std::string_view ss;

//   func(std::string(ss));
// }

// template <uint64_t, uint32_t, typename TARG, typename F>
// struct MemberFunctionWrapper;

// template <uint64_t H, uint32_t N, typename T, typename R, typename... Args>
// struct MemberFunctionWrapper<H, N, R (T::*)(Args...)> {
//   using return_type = R;
//   using func_t = R (T::*)(Args...);
//   static std::string& GetFuncName() {
//     static std::string func_name;
//     return func_name;
//   }
//   static func_t& GetFunc() {
//     static func_t func = nullptr;
//     return func;
//   }
//   static R Call(T* p, Args... args) {
//     auto func = GetFunc();
//     if constexpr (std::is_same_v<void, R>) {
//       (p->*func)(std::forward<Args>(args)...);
//     } else {
//       return (p->*func)(std::forward<Args>(args)...);
//     }
//   }
// };
// template <uint64_t H, uint32_t N, typename T, typename R, typename... Args>
// struct MemberFunctionWrapper<H, N, R (T::*)(Args...) const> {
//   using return_type = R;
//   using func_t = R (T::*)(Args...) const;
//   static std::string& GetFuncName() {
//     static std::string func_name;
//     return func_name;
//   }
//   static func_t& GetFunc() {
//     static func_t func = nullptr;
//     return func;
//   }
//   static R Call(const T* p, Args... args) {
//     auto func = GetFunc();
//     if constexpr (std::is_same_v<void, R>) {
//       (p->*func)(std::forward<Args>(args)...);
//     } else {
//       return (p->*func)(std::forward<Args>(args)...);
//     }
//   }
// };
}  // namespace rapidudf