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
#include <utility>
#include "boost/parser/parser.hpp"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace ast {

class Symbols {
 public:
  static boost::parser::symbols<std::pair<DType, DTypeAttr>> kDtypeSymbols;
  static boost::parser::symbols<DType> kNumberSymbols;
  static boost::parser::symbols<OpToken> kAssignOpSymbols;
  static boost::parser::symbols<OpToken> kLogicOpSymbols;
  static boost::parser::symbols<OpToken> kCmpOpSymbols;
  static boost::parser::symbols<OpToken> kAdditiveOpSymbols;
  static boost::parser::symbols<OpToken> kMultiplicativeOpSymbols;
  static boost::parser::symbols<OpToken> kPowerOpSymbols;
  static boost::parser::symbols<OpToken> kUnaryOpSymbols;
  static boost::parser::symbols<uint32_t> kContinueSymbols;
  static boost::parser::symbols<uint32_t> kBreakSymbols;

  static void Init();
  Symbols();

 private:
  static void Add(const std::string& name, DType dtype, DTypeAttr attr);
};

}  // namespace ast
}  // namespace rapidudf