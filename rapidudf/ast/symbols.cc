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

#include "rapidudf/ast/symbols.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"

namespace rapidudf {
namespace ast {

// Cache to keep string keys alive for the lifetime of the program
using SymbolCache = std::unordered_map<std::string, std::unique_ptr<std::string>>;
static SymbolCache& get_symbol_token_cache() {
  static SymbolCache c;
  return c;
}

static const std::string& intern(const std::string& name) {
  auto& cache = get_symbol_token_cache();
  auto found = cache.find(name);
  if (found != cache.end()) {
    return *found->second;
  }
  auto p = std::make_unique<std::string>(name);
  const std::string& ref = *p;
  cache.emplace(name, std::move(p));
  return ref;
}

const std::unordered_map<std::string, DType>& Symbols::GetNumberSymbols() {
  static const std::unordered_map<std::string, DType> m = {
      {"u8", DType(DATA_U8)},   {"i8", DType(DATA_I8)},   {"u16", DType(DATA_U16)}, {"i16", DType(DATA_I16)},
      {"u32", DType(DATA_U32)}, {"i32", DType(DATA_U32)}, {"u64", DType(DATA_U64)}, {"i64", DType(DATA_I64)},
      {"f32", DType(DATA_F32)}, {"f64", DType(DATA_F64)}, {"f80", DType(DATA_F80)}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetAssignOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"=", OP_ASSIGN},         {"+=", OP_PLUS_ASSIGN},
                                                              {"-=", OP_MINUS_ASSIGN},  {"*=", OP_MULTIPLY_ASSIGN},
                                                              {"/=", OP_DIVIDE_ASSIGN}, {"%=", OP_MOD_ASSIGN}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetLogicOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"||", OP_LOGIC_OR}, {"&&", OP_LOGIC_AND}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetCmpOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"==", OP_EQUAL},         {"!=", OP_NOT_EQUAL},
                                                              {">=", OP_GREATER_EQUAL}, {"<=", OP_LESS_EQUAL},
                                                              {">", OP_GREATER},        {"<", OP_LESS}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetAdditiveOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"+", OP_PLUS}, {"-", OP_MINUS}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetMultiplicativeOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {
      {"*", OP_MULTIPLY}, {"/", OP_DIVIDE}, {"%", OP_MOD}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetPowerOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"^", OP_POW}};
  return m;
}

const std::unordered_map<std::string, OpToken>& Symbols::GetUnaryOpSymbols() {
  static const std::unordered_map<std::string, OpToken> m = {{"-", OP_NEGATIVE}, {"!", OP_NOT}};
  return m;
}

std::mutex& Symbols::GetDtypeSymbolsMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, std::pair<DType, DTypeAttr>>& Symbols::GetDtypeSymbols() {
  static std::unordered_map<std::string, std::pair<DType, DTypeAttr>> m;
  return m;
}

bool Symbols::IsDTypeExist(const std::string& id) {
  std::lock_guard<std::mutex> guard(GetDtypeSymbolsMutex());
  return GetDtypeSymbols().count(id) > 0;
}

std::optional<std::pair<DType, DTypeAttr>> Symbols::FindDType(const std::string& id) {
  std::lock_guard<std::mutex> guard(GetDtypeSymbolsMutex());
  auto& m = GetDtypeSymbols();
  auto it = m.find(id);
  if (it != m.end()) {
    return it->second;
  }
  return std::nullopt;
}

void Symbols::Add(const std::string& name, DType dtype, DTypeAttr attr) {
  std::lock_guard<std::mutex> guard(GetDtypeSymbolsMutex());
  intern(name);
  auto& m = GetDtypeSymbols();
  if (dtype.IsPtr()) {
    m[name] = {dtype, attr};
  } else if (!dtype.IsPrimitive()) {
    DType reg_dtype = dtype;
    if (!dtype.IsSimdVector()) {
      reg_dtype = dtype.ToPtr();
    }
    m[name] = {reg_dtype, attr};
  }
}

void Symbols::Init() {
  std::lock_guard<std::mutex> guard(GetDtypeSymbolsMutex());
  auto& m = GetDtypeSymbols();
  static const DTypeAttr empty_attr;

  // Built-in types
  m["void"] = {DType(DATA_VOID), empty_attr};
  m["bool"] = {DType(DATA_BIT), empty_attr};
  m["u8"] = {DType(DATA_U8), empty_attr};
  m["i8"] = {DType(DATA_I8), empty_attr};
  m["u16"] = {DType(DATA_U16), empty_attr};
  m["i16"] = {DType(DATA_I16), empty_attr};
  m["u32"] = {DType(DATA_U32), empty_attr};
  m["i32"] = {DType(DATA_I32), empty_attr};
  m["u64"] = {DType(DATA_U64), empty_attr};
  m["i64"] = {DType(DATA_I64), empty_attr};
  m["f16"] = {DType(DATA_F16), empty_attr};
  m["f32"] = {DType(DATA_F32), empty_attr};
  m["f64"] = {DType(DATA_F64), empty_attr};
  m["f80"] = {DType(DATA_F80), empty_attr};
  m["int"] = {DType(DATA_I32), empty_attr};
  m["long"] = {DType(DATA_I64), empty_attr};
  m["float"] = {DType(DATA_F32), empty_attr};
  m["double"] = {DType(DATA_F64), empty_attr};
  m["json"] = {DType(DATA_JSON).ToPtr(), empty_attr};
  m["string"] = {DType(DATA_STRING).ToPtr(), empty_attr};
  m["std_string_view"] = {DType(DATA_STD_STRING_VIEW), empty_attr};
  m["Context"] = {DType(DATA_CONTEXT).ToPtr(), empty_attr};
  m["string_view"] = {DType(DATA_STRING_VIEW), empty_attr};

  // Register types from DTypeFactory
  DTypeFactory::Visit([](const std::string& name, DType dtype) {
    auto& m = GetDtypeSymbols();
    static const DTypeAttr empty_attr;
    intern(name);
    if (dtype.IsPtr()) {
      m[name] = {dtype, empty_attr};
    } else if (!dtype.IsPrimitive()) {
      DType reg_dtype = dtype;
      if (!dtype.IsSimdVector()) {
        reg_dtype = dtype.ToPtr();
      }
      m[name] = {reg_dtype, empty_attr};
    }
  });

  // Register DynObjectSchema types
  auto schema_names = DynObjectSchema::ListAll();
  DType dyn_obj_dtype = DType(DATA_DYN_OBJECT).ToPtr();
  for (auto& name : schema_names) {
    std::string obj_name;
    if (DynObjectSchema::Get(name)->IsTable()) {
      obj_name = fmt::format("{}<{}>", kFundamentalTypeStrs[DATA_TABLE], name);
    } else {
      obj_name = fmt::format("{}<{}>", kFundamentalTypeStrs[DATA_DYN_OBJECT], name);
    }
    DTypeAttr attr;
    attr.schema = name;
    // Use intern to keep the name alive
    const std::string& stable_name = intern(obj_name);
    m[stable_name] = {dyn_obj_dtype, attr};
  }
}

}  // namespace ast
}  // namespace rapidudf
