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

#include "rapidudf/ast/grammar.h"
#include <chrono>
#include <unordered_set>
#include "boost/parser/parser.hpp"
#include "fmt/core.h"

#include "rapidudf/ast/block.h"
#include "rapidudf/ast/expression.h"
#include "rapidudf/ast/function.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/ast/symbols.h"

namespace rapidudf {
namespace ast {
namespace bp = boost::parser;
using namespace bp::literals;

static std::unordered_set<std::string> g_keywords = {"return", "if", "elif", "else", "while"};

auto identifier_func = [](auto& ctx) {
  const std::string& id = _attr(ctx);
  // auto found = Symbols::kDtypeSymbols.find(ctx, id);
  if (!Symbols::IsDTypeExist(ctx, id)) {
    std::string err_msg = fmt::format("invalid identiier:{} which is reserved dtype name.", id);
    _report_error(ctx, err_msg);
    _pass(ctx) = false;
    return;
  }

  if (g_keywords.count(id) == 1) {
    std::string err_msg = fmt::format("invalid identiier:{} which is keyword", id);
    _report_error(ctx, err_msg);
    _pass(ctx) = false;
    return;
  }

  _val(ctx) = _attr(ctx);
};

bp::rule<struct comment> comment = "comment";
const auto comment_def = "//" > *(bp::char_ - bp::eol) > (bp::eol);

bp::rule<struct identifier, std::string> identifier = "identifier";
const auto lead_char = bp::char_('a', 'z') | bp::char_('A', 'Z') | bp::char_('_');
const auto follow_char =
    bp::char_('a', 'z') | bp::char_('A', 'Z') | bp::char_('_') | bp::char_('0', '9') | bp::char_(':');
const auto identifier_def = (lead_char > *follow_char)[identifier_func];
BOOST_PARSER_DEFINE_RULES(identifier);

bp::rule<struct operand, Operand> operand = "operand";
bp::rule<struct assign, BinaryExprPtr> assign = "assign";
bp::rule<struct logic_expr, BinaryExprPtr> logic_expr = "logic_expr";
bp::rule<struct cmp_expr, BinaryExprPtr> cmp_expr = "cmp_expr";
bp::rule<struct additive_expr, BinaryExprPtr> additive_expr = "additive_expr";
bp::rule<struct multiplicative_expr, BinaryExprPtr> multiplicative_expr = "multiplicative_expr";
bp::rule<struct power_expr, BinaryExprPtr> power_expr = "power_expr";
bp::rule<struct unary_expr, UnaryExprPtr> unary_expr = "unary_expr";
bp::rule<struct ternary_expr, SelectExprPtr> ternary_expr = "ternary_expr";
bp::rule<struct expression, BinaryExprPtr> expression = "expression";
bp::rule<struct func_invoke_args, FuncInvokeArgs> func_invoke_args = "func_invoke_args";

bp::rule<struct var_ref, VarRef> var_ref = "var_ref";
bp::rule<struct var_declare, VarDefine> var_declare = "var_declare";
bp::rule<struct filed_access, FieldAccess> filed_access = "filed_access";
bp::rule<struct dynamic_param_access, DynamicParamAccess> dynamic_param_access = "dynamic_param_access";
bp::rule<struct member_access, std::vector<MemberAccess>> member_access = "member_access";
bp::rule<struct var_accessor, VarAccessor> var_accessor = "var_accessor";
bp::rule<struct constant_number, ConstantNumber> constant_number = "constant_number";
bp::rule<struct array, Array> array = "array";

auto func_convert = [](auto& ctx) {
  Function f;
  f.return_type = std::get<0>(_attr(ctx)).first;
  f.name = std::get<1>(_attr(ctx));
  f.args = std::get<2>(_attr(ctx));
  f.body = std::get<3>(_attr(ctx));
  f.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = f;
};

auto var_declare_func = [](auto& ctx) {
  VarDefine v;
  v.name = _attr(ctx);
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto var_ref_func = [](auto& ctx) {
  VarRef v;
  v.name = _attr(ctx);
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto unary_expr_func = [](auto& ctx) {
  auto v = std::make_shared<UnaryExpr>();
  v->op = std::get<0>(_attr(ctx));
  v->operand = std::get<1>(_attr(ctx));
  v->position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto binary_expr_func = [](auto& ctx) {
  auto v = std::make_shared<BinaryExpr>();
  v->left = std::get<0>(_attr(ctx));
  // v->right = std::get<1>(_attr(ctx));
  v->SetRight(std::get<1>(_attr(ctx)));
  v->position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto ternary_expr_func = [](auto& ctx) {
  auto v = std::make_shared<SelectExpr>();
  v->cond = std::get<0>(_attr(ctx));
  v->true_false_operands = std::get<1>(_attr(ctx));
  v->position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto var_accessor_func = [](auto& ctx) {
  VarAccessor f;
  f.name = std::get<0>(_attr(ctx));
  auto args = std::get<1>(_attr(ctx));
  if (args.has_value()) {
    if (args->index() == 0) {
      f.access_args = std::get<0>(*args);
    } else if (args->index() == 1) {
      f.func_args = std::get<1>(*args);
    }
  }
  f.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = f;
};
auto array_func = [](auto& ctx) {
  Array v;
  v.elements = _attr(ctx);
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};

auto field_access_func = [](auto& ctx) {
  FieldAccess v;
  v.field = std::get<0>(_attr(ctx));
  v.func_args = std::get<1>(_attr(ctx));
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};

auto continue_func = [](auto& ctx) {
  ContinueStatement v;
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};
auto break_func = [](auto& ctx) {
  BreakStatement v;
  v.position = _where(ctx).begin() - _begin(ctx);
  _val(ctx) = v;
};

auto expr_stmt_func = [](auto& ctx) {
  ExpressionStatement v;
  v.expr = _attr(ctx);
  _val(ctx) = v;
};

auto return_stmt_func = [](auto& ctx) {
  ReturnStatement v;
  v.expr = _attr(ctx);
  _val(ctx) = v;
};

auto choice_stmt_func = [](auto& ctx) {
  ChoiceStatement v;
  v.expr = std::get<0>(_attr(ctx));
  v.statements = std::get<1>(_attr(ctx));
  _val(ctx) = v;
};

auto ifelse_stmt_func = [](auto& ctx) {
  IfElseStatement v;
  // v.if_statement = std::get<0>(_attr(ctx));
  v.elif_statements = std::get<0>(_attr(ctx));
  v.else_statements = std::get<1>(_attr(ctx));
  _val(ctx) = v;
};

auto func_invoke_args_func = [](auto& ctx) {
  FuncInvokeArgs f;
  f.args = _attr(ctx);
  _val(ctx) = f;
};

auto func_arg_func = [](auto& ctx) {
  FunctionArg f;
  auto dtype_attr = std::get<0>(_attr(ctx));
  f.dtype = dtype_attr.first;
  f.attr = dtype_attr.second;
  f.name = std::get<1>(_attr(ctx));
  _val(ctx) = f;
};

auto const constant_number_def = bp::lexeme[bp::double_ > -('_' > Symbols::kNumberSymbols)];
auto const var_declare_def = ("auto" > identifier)[var_declare_func];
auto const var_ref_def = identifier[var_ref_func];
auto const array_def = ('[' > (expression % ',') > ']')[array_func];
auto const operand_def =
    constant_number | bp::bool_ | bp::quoted_string | var_declare | var_accessor | ('(' >> expression >> ')') | array;
auto const func_invoke_args_def = ('(' > -(expression % ',') > ')')[func_invoke_args_func];
auto const expression_def = assign;
auto const assign_def = (ternary_expr >> -(Symbols::kAssignOpSymbols >> expression))[binary_expr_func];
auto const ternary_expr_def = (logic_expr > -('?' > expression > ':' > expression))[ternary_expr_func];
auto const logic_expr_def = (cmp_expr >> *(Symbols::kLogicOpSymbols >> cmp_expr))[binary_expr_func];
auto const cmp_expr_def = (additive_expr >> *(Symbols::kCmpOpSymbols >> additive_expr))[binary_expr_func];
auto const additive_expr_def =
    (multiplicative_expr >> *(Symbols::kAdditiveOpSymbols >> multiplicative_expr))[binary_expr_func];
auto const multiplicative_expr_def =
    (power_expr >> *(Symbols::kMultiplicativeOpSymbols >> power_expr))[binary_expr_func];
auto const power_expr_def = (unary_expr >> *(Symbols::kPowerOpSymbols >> unary_expr))[binary_expr_func];
auto const unary_expr_def = (-Symbols::kUnaryOpSymbols >> operand)[unary_expr_func];

auto const filed_access_def = (('.' > identifier > -func_invoke_args))[field_access_func];
auto const dynamic_param_access_def = ('[' > (bp::quoted_string | bp::uint_ | var_ref) > ']');
auto const member_access_def = +(filed_access | dynamic_param_access);
auto const var_accessor_def = (identifier > -(member_access | func_invoke_args))[var_accessor_func];

BOOST_PARSER_DEFINE_RULES(constant_number, var_declare, var_ref, var_accessor, filed_access, dynamic_param_access,
                          operand, func_invoke_args, member_access, unary_expr, assign, logic_expr, cmp_expr,
                          additive_expr, multiplicative_expr, power_expr, expression, ternary_expr, array);

bp::rule<struct statements, std::vector<Statement>> statements = "statements";
bp::rule<struct return_statement, ReturnStatement> return_statement = "return_statement";
bp::rule<struct expr_statement, ExpressionStatement> expr_statement = "expr_statement";
bp::rule<struct while_statement, WhileStatement> while_statement = "while_statement";
bp::rule<struct ifelse_statement, IfElseStatement> ifelse_statement = "ifelse_statement";
bp::rule<struct choice_statement, ChoiceStatement> choice_statement = "choice_statement";
bp::rule<struct continue_statement, ContinueStatement> continue_statement = "continue_statement";
bp::rule<struct break_statement, BreakStatement> break_statement = "break_statement";
bp::rule<struct block, Block> block = "block";

auto const statements_def =
    *(continue_statement | break_statement | return_statement | while_statement | ifelse_statement | expr_statement);
auto const return_statement_def = ("return" > -expression > ';')[return_stmt_func];
auto const continue_statement_def = (Symbols::kContinueSymbols > ';')[continue_func];
auto const break_statement_def = (Symbols::kBreakSymbols > ';')[break_func];
auto const expr_statement_def = (expression > ';')[expr_stmt_func];
auto const while_statement_def = "while" > choice_statement;
auto const choice_statement_def = ('(' > expression > ')' > '{' > statements > '}')[choice_stmt_func];
auto const ifelse_statement_def = ("if" > choice_statement > *("elif" > choice_statement) >
                                   -("else" > *bp::ws > '{' > statements > '}'))[ifelse_stmt_func];
auto const block_def = '{' > statements > '}';

bp::rule<struct func_arg, FunctionArg> func_arg = "func_arg";
auto const func_arg_def = (Symbols::GetDtypeSymbols() > identifier)[func_arg_func];
bp::rule<struct func_args, std::vector<FunctionArg>> func_args = "func_args";
auto const func_args_def = func_arg % ',';
bp::rule<struct func, Function> func = "func";
bp::rule<struct funcs, std::vector<Function>> funcs = "funcs";
auto const func_def = (Symbols::GetDtypeSymbols() > identifier > '(' > -func_args >> ')' > block)[func_convert];
auto const funcs_def = +func;
BOOST_PARSER_DEFINE_RULES(comment, block, func_arg, func_args, func, funcs, return_statement, statements,
                          expr_statement, while_statement, choice_statement, ifelse_statement, break_statement,
                          continue_statement);

absl::StatusOr<Function> parse_function_ast(ParseContext& ctx, const std::string& source) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source);
  bp::callback_error_handler error_handler([&](std::string const& msg) { ctx.SetAstErr(msg); });
  auto const parser = bp::with_error_handler(func, error_handler);
  std::optional<Function> result = bp::parse(source, parser, bp::ws | comment);
  auto parse_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  start_time = std::chrono::high_resolution_clock::now();
  ctx.SetParseCost(parse_duration);
  if (result) {
    ctx.SetFuncDesc(result->ToFuncDesc());
    auto rc = result->Validate(ctx);
    if (!rc.ok()) {
      return rc;
    }
    ctx.SetParseValidateCost(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  }
  if (!result) {
    return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, ctx.GetAstErr()));
  }
  return *result;
}

absl::StatusOr<std::vector<Function>> parse_functions_ast(ParseContext& ctx, const std::string& source) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source);
  bp::callback_error_handler error_handler([&](std::string const& msg) { ctx.SetAstErr(msg); });
  auto const parser = bp::with_error_handler(funcs, error_handler);
  std::optional<std::vector<Function>> result = bp::parse(source, parser, bp::ws | comment);
  ctx.SetParseCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  start_time = std::chrono::high_resolution_clock::now();
  if (result) {
    std::set<std::string> func_names;
    for (size_t i = 0; i < result->size(); i++) {
      ctx.SetFunctionCursor(i);
      ctx.SetFuncDesc(result->at(i).ToFuncDesc(), i);
      auto rc = result->at(i).Validate(ctx);
      if (!rc.ok()) {
        return rc;
      }

      if (!func_names.insert(result->at(i).name).second) {
        return absl::InvalidArgumentError(fmt::format("duplicate function name:{}", result->at(i).name));
      }
    }
  }
  ctx.SetParseValidateCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  if (!result) {
    return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, ctx.GetAstErr()));
  }
  return *result;
}

absl::StatusOr<Expression> parse_expression_ast(ParseContext& ctx, const std::string& source,
                                                const FunctionDesc& desc) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source, false);
  bp::callback_error_handler error_handler([&](std::string const& msg) { ctx.SetAstErr(msg); });
  auto const parser = bp::with_error_handler(expression, error_handler);
  std::optional<BinaryExprPtr> result = bp::parse(source, parser, bp::ws | comment);
  ctx.SetParseCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  start_time = std::chrono::high_resolution_clock::now();
  Expression expr;
  if (result) {
    ctx.SetFuncDesc(desc);
    expr.expr = *result;
    auto rc = (*result)->Validate(ctx, expr.rpn_expr);
    if (!rc.ok()) {
      return rc.status();
    }
    ctx.SetParseValidateCost(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  }

  if (!result) {
    return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, ctx.GetAstErr()));
  }
  return expr;
}

}  // namespace ast
}  // namespace rapidudf