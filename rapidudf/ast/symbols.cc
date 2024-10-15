/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/ast/symbols.h"
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/dyn_object.h"
#include "rapidudf/types/dyn_object_schema.h"
namespace rapidudf {
namespace ast {
using SymbolCache = std::unordered_map<std::string, std::unique_ptr<std::string>>;
static SymbolCache& get_symbol_token_cache() {
  static SymbolCache c;
  return c;
}

boost::parser::symbols<DType> Symbols::kNumberSymbols = {
    {"u8", DType(DATA_U8)},   {"i8", DType(DATA_I8)},   {"u16", DType(DATA_U16)}, {"i16", DType(DATA_I16)},
    {"u32", DType(DATA_U32)}, {"i32", DType(DATA_U32)}, {"u64", DType(DATA_U64)}, {"i64", DType(DATA_I64)},
    {"f32", DType(DATA_F32)}, {"f64", DType(DATA_F64)}, {"f80", DType(DATA_F80)}};

static const DTypeAttr empty_attr;
boost::parser::symbols<std::pair<DType, DTypeAttr>> Symbols::kDtypeSymbols = {
    {"void", {DType(DATA_VOID), empty_attr}},
    {"bool", {DType(DATA_BIT), empty_attr}},
    {"u8", {DType(DATA_U8), empty_attr}},
    {"i8", {DType(DATA_I8), empty_attr}},
    {"u16", {DType(DATA_U16), empty_attr}},
    {"i16", {DType(DATA_I16), empty_attr}},
    {"u32", {DType(DATA_U32), empty_attr}},
    {"i32", {DType(DATA_I32), empty_attr}},
    {"u64", {DType(DATA_U64), empty_attr}},
    {"i64", {DType(DATA_I64), empty_attr}},
    {"f16", {DType(DATA_F16), empty_attr}},
    {"f32", {DType(DATA_F32), empty_attr}},
    {"f64", {DType(DATA_F64), empty_attr}},
    {"f80", {DType(DATA_F80), empty_attr}},
    {"int", {DType(DATA_I32), empty_attr}},
    {"long", {DType(DATA_I64), empty_attr}},
    {"float", {DType(DATA_F32), empty_attr}},
    {"double", {DType(DATA_F64), empty_attr}},
    {"json", {DType(DATA_JSON).ToPtr(), empty_attr}},
    {"string", {DType(DATA_STRING).ToPtr(), empty_attr}},
    {"std_string_view", {DType(DATA_STD_STRING_VIEW), empty_attr}},
    {"Context", {DType(DATA_CONTEXT).ToPtr(), empty_attr}},
    // {"dyn_obj", DType(DATA_DYN_OBJECT).ToPtr()},
    {"string_view", {DType(DATA_STRING_VIEW), empty_attr}}};

boost::parser::symbols<OpToken> Symbols::kAssignOpSymbols = {{"=", OP_ASSIGN},         {"+=", OP_PLUS_ASSIGN},
                                                             {"-=", OP_MINUS_ASSIGN},  {"*=", OP_MULTIPLY_ASSIGN},
                                                             {"/=", OP_DIVIDE_ASSIGN}, {"+=", OP_MOD_ASSIGN}};
boost::parser::symbols<OpToken> Symbols::kLogicOpSymbols = {{"||", OP_LOGIC_OR}, {"&&", OP_LOGIC_AND}};
boost::parser::symbols<OpToken> Symbols::kCmpOpSymbols = {{"==", OP_EQUAL},         {"!=", OP_NOT_EQUAL},
                                                          {">=", OP_GREATER_EQUAL}, {"<=", OP_LESS_EQUAL},
                                                          {">", OP_GREATER},        {"<", OP_LESS}};
boost::parser::symbols<OpToken> Symbols::kAdditiveOpSymbols = {{"+", OP_PLUS}, {"-", OP_MINUS}};
boost::parser::symbols<OpToken> Symbols::kMultiplicativeOpSymbols = {
    {"*", OP_MULTIPLY}, {"/", OP_DIVIDE}, {"%", OP_MOD}};
boost::parser::symbols<OpToken> Symbols::kPowerOpSymbols = {{"^", OP_POW}};
boost::parser::symbols<OpToken> Symbols::kUnaryOpSymbols = {{"-", OP_NEGATIVE}, {"!", OP_NOT}};
boost::parser::symbols<uint32_t> Symbols::kContinueSymbols = {{"continue", 0}};
boost::parser::symbols<uint32_t> Symbols::kBreakSymbols = {{"break", 0}};
// void Symbols::AddDType(const std::string& name, DType dtype) { kDtypeSymbols.insert_for_next_parse(name, dtype); }

void Symbols::Add(const std::string& name, DType dtype, DTypeAttr attr) {
  std::string_view name_view;
  auto& symbol_cache = get_symbol_token_cache();
  auto found = symbol_cache.find(name);
  if (found != symbol_cache.end()) {
    name_view = *found->second;
    return;
  } else {
    std::unique_ptr<std::string> name_str = std::make_unique<std::string>(name);
    name_view = *name_str;
    symbol_cache.emplace(name, std::move(name_str));
  }
  if (dtype.IsPtr()) {
    kDtypeSymbols.insert_for_next_parse(name_view, {dtype, attr});
  } else if (!dtype.IsPrimitive()) {
    DType reg_dtype = dtype;
    if (!dtype.IsSimdVector()) {
      reg_dtype = dtype.ToPtr();
    }
    kDtypeSymbols.insert_for_next_parse(name_view, {reg_dtype, attr});
  }
}
void Symbols::Init() {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  DTypeFactory::Visit([](const std::string& name, DType dtype) {
    std::string_view name_view;
    auto& symbol_cache = get_symbol_token_cache();
    auto found = symbol_cache.find(name);
    if (found != symbol_cache.end()) {
      name_view = *found->second;
      return;
    } else {
      std::unique_ptr<std::string> name_str = std::make_unique<std::string>(name);
      name_view = *name_str;
      symbol_cache.emplace(name, std::move(name_str));
    }
    if (dtype.IsPtr()) {
      kDtypeSymbols.insert_for_next_parse(name_view, {dtype, empty_attr});
    } else if (!dtype.IsPrimitive()) {
      DType reg_dtype = dtype;
      if (!dtype.IsSimdVector()) {
        reg_dtype = dtype.ToPtr();
      }
      kDtypeSymbols.insert_for_next_parse(name_view, {reg_dtype, empty_attr});
    }
  });

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
    Add(obj_name, dyn_obj_dtype, attr);
  }
}
}  // namespace ast
}  // namespace rapidudf