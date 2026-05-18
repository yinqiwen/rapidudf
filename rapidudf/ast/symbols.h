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
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace ast {

class Symbols {
 public:
  static const std::unordered_map<std::string_view, DType>& GetNumberSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetAssignOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetLogicOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetCmpOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetAdditiveOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetMultiplicativeOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetPowerOpSymbols();
  static const std::unordered_map<std::string_view, OpToken>& GetUnaryOpSymbols();

  static void Init();

  static bool IsDTypeExist(std::string_view id);
  static std::optional<std::pair<DType, DTypeAttr>> FindDType(std::string_view id);

  static void Add(const std::string& name, DType dtype, DTypeAttr attr);

 private:
  static std::unordered_map<std::string, std::pair<DType, DTypeAttr>>& GetDtypeSymbols();
  static std::mutex& GetDtypeSymbolsMutex();
};

}  // namespace ast
}  // namespace rapidudf
