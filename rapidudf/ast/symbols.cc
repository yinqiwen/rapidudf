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
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace ast {
static std::vector<std::unique_ptr<std::string>>& get_symbol_token_cache() {
  static std::vector<std::unique_ptr<std::string>> c;
  return c;
}

boost::parser::symbols<DType> Symbols::kNumberSymbols = {
    {"u8", DType(DATA_U8)},   {"i8", DType(DATA_I8)},   {"u16", DType(DATA_U16)}, {"i16", DType(DATA_I16)},
    {"u32", DType(DATA_U32)}, {"i32", DType(DATA_U32)}, {"u64", DType(DATA_U64)}, {"i64", DType(DATA_I64)},
    {"f32", DType(DATA_F32)}, {"f64", DType(DATA_F64)}};

boost::parser::symbols<DType> Symbols::kDtypeSymbols = {{"void", DType(DATA_VOID)},
                                                        {"bool", DType(DATA_U8)},
                                                        {"u8", DType(DATA_U8)},
                                                        {"i8", DType(DATA_I8)},
                                                        {"u16", DType(DATA_U16)},
                                                        {"i16", DType(DATA_I16)},
                                                        {"u32", DType(DATA_U32)},
                                                        {"i32", DType(DATA_U32)},
                                                        {"u64", DType(DATA_U64)},
                                                        {"i64", DType(DATA_I64)},
                                                        {"f32", DType(DATA_F32)},
                                                        {"f64", DType(DATA_F64)},
                                                        {"int", DType(DATA_I32)},
                                                        {"long", DType(DATA_I64)},
                                                        {"float", DType(DATA_F32)},
                                                        {"double", DType(DATA_F64)},
                                                        {"json", DType(DATA_JSON).ToPtr()},
                                                        {"string", DType(DATA_STRING).ToPtr()},
                                                        {"string_view", DType(DATA_STRING_VIEW)}};

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
boost::parser::symbols<OpToken> Symbols::kUnaryOpSymbols = {{"-", OP_NEGATIVE}, {"!", OP_NOT}};

// void Symbols::AddDType(const std::string& name, DType dtype) { kDtypeSymbols.insert_for_next_parse(name, dtype); }
void Symbols::Init() {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  DTypeFactory::Visit([](const std::string& name, DType dtype) {
    std::unique_ptr<std::string> name_str = std::make_unique<std::string>(name);
    std::string_view name_view = *name_str;
    get_symbol_token_cache().emplace_back(std::move(name_str));
    if (dtype.IsPtr()) {
      kDtypeSymbols.insert_for_next_parse(name_view, dtype);
    } else if (!dtype.IsPrimitive()) {
      DType reg_dtype = dtype;
      if (!dtype.IsSimdVector()) {
        reg_dtype = dtype.ToPtr();
      }
      kDtypeSymbols.insert_for_next_parse(name_view, reg_dtype);
      // RUDF_DEBUG("Add symbol {}:{}", name, reg_dtype);
    }
  });
  // kDtypeSymbols.insert_for_next_parse("vector<int>", DType(DATA_I32).ToVector().ToPtr());
}
}  // namespace ast
}  // namespace rapidudf