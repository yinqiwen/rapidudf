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

#include "absl/container/flat_hash_map.h"
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
  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const { return absl::Hash<std::string_view>()(sv); }
  };
  struct StringEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const { return a == b; }
  };
  using DtypeMap = absl::flat_hash_map<std::string, std::pair<DType, DTypeAttr>, StringHash, StringEq>;

  static DtypeMap& GetDtypeSymbols();
  static std::mutex& GetDtypeSymbolsMutex();
  static std::once_flag& GetInitFlag();
};

}  // namespace ast
}  // namespace rapidudf
