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
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <unordered_set>
#include "fmt/core.h"

#include "rapidudf/ast/ast_pool.h"
#include "rapidudf/ast/block.h"
#include "rapidudf/ast/expression.h"
#include "rapidudf/ast/function.h"
#include "rapidudf/ast/lexer.h"
#include "rapidudf/ast/statement.h"
#include "rapidudf/ast/symbols.h"

namespace rapidudf {
namespace ast {

// ---- Identifier validation ----

static const std::unordered_set<std::string_view>& GetReservedKeywords() {
  static const std::unordered_set<std::string_view> keywords = {
      "return", "if", "elif", "else", "while", "auto", "break", "continue"};
  return keywords;
}

static bool IsValidIdentifier(std::string_view name) {
  if (Symbols::IsDTypeExist(name)) {
    return false;
  }
  if (GetReservedKeywords().count(name) > 0) {
    return false;
  }
  return true;
}

// ---- Operator mapping helpers ----

static std::optional<OpToken> AssignOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_ASSIGN:         return OP_ASSIGN;
    case TOKEN_PLUS_ASSIGN:    return OP_PLUS_ASSIGN;
    case TOKEN_MINUS_ASSIGN:   return OP_MINUS_ASSIGN;
    case TOKEN_STAR_ASSIGN:    return OP_MULTIPLY_ASSIGN;
    case TOKEN_SLASH_ASSIGN:   return OP_DIVIDE_ASSIGN;
    case TOKEN_PERCENT_ASSIGN: return OP_MOD_ASSIGN;
    default: return std::nullopt;
  }
}

static std::optional<OpToken> LogicOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_OR:  return OP_LOGIC_OR;
    case TOKEN_AND: return OP_LOGIC_AND;
    default: return std::nullopt;
  }
}

static std::optional<OpToken> CmpOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_EQ: return OP_EQUAL;
    case TOKEN_NE: return OP_NOT_EQUAL;
    case TOKEN_GE: return OP_GREATER_EQUAL;
    case TOKEN_LE: return OP_LESS_EQUAL;
    case TOKEN_GT: return OP_GREATER;
    case TOKEN_LT: return OP_LESS;
    default: return std::nullopt;
  }
}

static std::optional<OpToken> AdditiveOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_PLUS:  return OP_PLUS;
    case TOKEN_MINUS: return OP_MINUS;
    default: return std::nullopt;
  }
}

static std::optional<OpToken> MultiplicativeOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_STAR:    return OP_MULTIPLY;
    case TOKEN_SLASH:   return OP_DIVIDE;
    case TOKEN_PERCENT: return OP_MOD;
    default: return std::nullopt;
  }
}

static std::optional<OpToken> UnaryOpFromToken(TokenType t) {
  switch (t) {
    case TOKEN_MINUS: return OP_NEGATIVE;
    case TOKEN_NOT:   return OP_NOT;
    default: return std::nullopt;
  }
}

// ---- Parser context ----

struct Parser {
  Lexer& lex;
  ParseContext& ctx;

  uint32_t Pos(const Token& t) { return static_cast<uint32_t>(t.position); }

  // Parse a double value from a number token, optionally extracting a type suffix
  bool ParseNumber(const Token& tok, double& val, std::optional<DType>& dtype) {
    std::string_view text = tok.text;
    size_t num_len = text.size();

    // Check for type suffix like _f32, _i32
    for (size_t i = 1; i < text.size(); ++i) {
      if (text[i] == '_') {
        num_len = i;
        std::string_view suffix = text.substr(i + 1);
        auto& num_syms = Symbols::GetNumberSymbols();
        auto it = num_syms.find(suffix);
        if (it != num_syms.end()) {
          dtype = it->second;
        } else {
          lex.SetError(fmt::format("unknown number suffix: {}", suffix));
          return false;
        }
        break;
      }
    }

    // Parse the numeric part using from_chars (no std::string allocation)
    std::string_view num_part = text.substr(0, num_len);
    auto [ptr, ec] = std::from_chars(num_part.data(), num_part.data() + num_part.size(), val);
    if (ec != std::errc() || ptr != num_part.data() + num_part.size()) {
      lex.SetError(fmt::format("invalid number: {}", num_part));
      return false;
    }
    return true;
  }

  // ---- Expression parsing (Pratt-style precedence climbing) ----

  // operand = constant_number | bool | string | var_declare | var_accessor | '(' expression ')' | array
  absl::StatusOr<Operand> parse_operand() {
    Token tok = lex.Peek();

    // Boolean literals
    if (tok.Is(TOKEN_BOOL)) {
      lex.Next();
      bool val = (tok.text == "true");
      return Operand(val);
    }

    // String literals
    if (tok.Is(TOKEN_STRING)) {
      lex.Next();
      // Strip quotes
      std::string s(tok.text.substr(1, tok.text.size() - 2));
      return Operand(s);
    }

    // Number
    if (tok.Is(TOKEN_NUMBER)) {
      lex.Next();
      double val;
      std::optional<DType> dtype;
      if (!ParseNumber(tok, val, dtype)) {
        return absl::InvalidArgumentError(lex.GetError());
      }
      ConstantNumber cn;
      cn.dv = val;
      cn.dtype = dtype;
      return Operand(cn);
    }

    // auto var_declare
    if (tok.Is(TOKEN_AUTO)) {
      lex.Next();
      Token name_tok = lex.Expect(TOKEN_IDENTIFIER);
      if (name_tok.Is(TOKEN_ERROR)) {
        return absl::InvalidArgumentError("expected identifier after 'auto'");
      }
      if (!IsValidIdentifier(name_tok.text)) {
        return absl::InvalidArgumentError(
            fmt::format("invalid identifier: '{}' is a reserved name", name_tok.text));
      }
      VarDefine vd;
      vd.name = name_tok.text;
      vd.position = Pos(name_tok);
      return Operand(vd);
    }

    // Parenthesized expression
    if (tok.Is(TOKEN_LPAREN)) {
      lex.Next();
      auto expr = parse_expression();
      if (!expr.ok()) return expr.status();
      Token rparen = lex.Expect(TOKEN_RPAREN);
      if (rparen.Is(TOKEN_ERROR)) {
        return absl::InvalidArgumentError("expected ')'");
      }
      return Operand(*expr);
    }

    // Array literal
    if (tok.Is(TOKEN_LBRACKET)) {
      return parse_array();
    }

    // Identifier: var_accessor
    if (tok.Is(TOKEN_IDENTIFIER)) {
      return parse_var_accessor();
    }

    return absl::InvalidArgumentError(
        fmt::format("unexpected token '{}' at position {}", tok.text, tok.position));
  }

  // array = '[' expression { ',' expression } ']'
  absl::StatusOr<Operand> parse_array() {
    lex.Next();  // consume '['
    Array arr;
    arr.position = Pos(lex.Peek());

    if (!lex.PeekIs(TOKEN_RBRACKET)) {
      auto elem = parse_expression();
      if (!elem.ok()) return elem.status();
      arr.elements.push_back(*elem);

      while (lex.ConsumeIf(TOKEN_COMMA)) {
        elem = parse_expression();
        if (!elem.ok()) return elem.status();
        arr.elements.push_back(*elem);
      }
    }

    Token rbracket = lex.Expect(TOKEN_RBRACKET);
    if (rbracket.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ']'");
    }
    return Operand(arr);
  }

  // var_accessor = identifier [ member_access | func_invoke_args ]
  absl::StatusOr<Operand> parse_var_accessor() {
    Token name_tok = lex.Next();  // identifier
    if (!IsValidIdentifier(name_tok.text)) {
      return absl::InvalidArgumentError(
          fmt::format("invalid identifier: '{}' is a reserved name", name_tok.text));
    }
    VarAccessor va;
    va.name = name_tok.text;
    va.position = Pos(name_tok);

    // Check for member access or function invoke args
    Token peek = lex.Peek();
    if (peek.Is(TOKEN_DOT) || peek.Is(TOKEN_LBRACKET)) {
      // member_access
      auto ma = parse_member_access();
      if (!ma.ok()) return ma.status();
      va.access_args = *ma;
    } else if (peek.Is(TOKEN_LPAREN)) {
      // func_invoke_args
      auto fia = parse_func_invoke_args();
      if (!fia.ok()) return fia.status();
      va.func_args = *fia;
    }

    return Operand(va);
  }

  // member_access = ( field_access | dynamic_param_access )+
  absl::StatusOr<std::vector<MemberAccess>> parse_member_access() {
    std::vector<MemberAccess> result;

    while (lex.PeekIs(TOKEN_DOT) || lex.PeekIs(TOKEN_LBRACKET)) {
      if (lex.PeekIs(TOKEN_DOT)) {
        auto fa = parse_field_access();
        if (!fa.ok()) return fa.status();
        result.push_back(*fa);
      } else {
        auto dpa = parse_dynamic_param_access();
        if (!dpa.ok()) return dpa.status();
        result.push_back(*dpa);
      }
    }
    return result;
  }

  // field_access = '.' identifier [ func_invoke_args ]
  absl::StatusOr<FieldAccess> parse_field_access() {
    lex.Next();  // consume '.'
    Token name_tok = lex.Expect(TOKEN_IDENTIFIER);
    if (name_tok.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected identifier after '.'");
    }
    FieldAccess fa;
    fa.field = name_tok.text;
    fa.position = Pos(name_tok);

    if (lex.PeekIs(TOKEN_LPAREN)) {
      auto fia = parse_func_invoke_args();
      if (!fia.ok()) return fia.status();
      fa.func_args = *fia;
    }
    return fa;
  }

  // dynamic_param_access = '[' ( string | uint | var_ref ) ']'
  absl::StatusOr<DynamicParamAccess> parse_dynamic_param_access() {
    lex.Next();  // consume '['

    Token tok = lex.Peek();
    DynamicParamAccess dpa;

    if (tok.Is(TOKEN_STRING)) {
      lex.Next();
      dpa = std::string(tok.text.substr(1, tok.text.size() - 2));
    } else if (tok.Is(TOKEN_NUMBER)) {
      lex.Next();
      uint32_t idx = 0;
      auto [ptr, ec] = std::from_chars(tok.text.data(), tok.text.data() + tok.text.size(), idx);
      (void)ptr;
      (void)ec;
      dpa = idx;
    } else if (tok.Is(TOKEN_IDENTIFIER)) {
      lex.Next();
      VarRef vr;
      vr.name = tok.text;
      vr.position = Pos(tok);
      dpa = vr;
    } else {
      return absl::InvalidArgumentError("expected string, number, or identifier in '[]'");
    }

    Token rbracket = lex.Expect(TOKEN_RBRACKET);
    if (rbracket.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ']'");
    }
    return dpa;
  }

  // func_invoke_args = '(' [ expression { ',' expression } ] ')'
  absl::StatusOr<FuncInvokeArgs> parse_func_invoke_args() {
    lex.Next();  // consume '('
    FuncInvokeArgs fia;

    if (!lex.PeekIs(TOKEN_RPAREN)) {
      std::vector<BinaryExprPtr> args;
      auto arg = parse_expression();
      if (!arg.ok()) return arg.status();
      args.push_back(*arg);

      while (lex.ConsumeIf(TOKEN_COMMA)) {
        arg = parse_expression();
        if (!arg.ok()) return arg.status();
        args.push_back(*arg);
      }
      fia.args = std::move(args);
    }

    Token rparen = lex.Expect(TOKEN_RPAREN);
    if (rparen.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ')'");
    }
    return fia;
  }

  // unary_expr = [ unary_op ] operand
  absl::StatusOr<UnaryExprPtr> parse_unary_expr() {
    Token peek = lex.Peek();
    std::optional<OpToken> unop;

    if (peek.Is(TOKEN_MINUS) || peek.Is(TOKEN_NOT)) {
      unop = UnaryOpFromToken(peek.type);
      lex.Next();
    }

    auto operand = parse_operand();
    if (!operand.ok()) return operand.status();

    auto* ue = ctx.GetAstPool().New<UnaryExpr>();
    ue->op = unop;
    ue->operand = std::move(*operand);
    ue->position = unop.has_value() ? Pos(peek) : 0;
    return UnaryExprPtr(ue);
  }

  // power_expr = unary_expr { '^' unary_expr }  (right-associative)
  // Since power is right-associative, we parse: unary_expr '^' power_expr | unary_expr
  // Note: The old Boost.Parser implementation used left-associative parsing for '^',
  // which was mathematically incorrect. This hand-written parser intentionally uses
  // right-associativity so that a^b^c = a^(b^c), which is the standard convention.
  absl::StatusOr<BinaryExprPtr> parse_power_expr() {
    auto left_ue = parse_unary_expr();
    if (!left_ue.ok()) return left_ue.status();

    if (lex.PeekIs(TOKEN_CARET)) {
      Token op_tok = lex.Next();
      // Right-associative: parse power_expr recursively
      auto right = parse_power_expr();
      if (!right.ok()) return right.status();

      auto be = BinaryExpr::New(ctx.GetAstPool(), std::move(*left_ue), Pos(op_tok));
      be->right.emplace_back(OP_POW, *right);
      return be;
    }

    // No power op, wrap unary as binary
    return BinaryExpr::New(ctx.GetAstPool(), std::move(*left_ue), 0);
  }

  // Helper: parse left-associative binary ops at a given precedence level
  // parse_next: function to parse the next-higher precedence level
  // op_match: function to check if current token is an operator at this level
  template <typename ParseNext, typename OpMatch>
  absl::StatusOr<BinaryExprPtr> parse_binary_left(ParseNext parse_next, OpMatch op_match) {
    auto left = parse_next();
    if (!left.ok()) return left.status();

    while (true) {
      Token peek = lex.Peek();
      auto op = op_match(peek.type);
      if (!op.has_value()) break;
      lex.Next();

      auto right = parse_next();
      if (!right.ok()) return right.status();

      auto be = BinaryExpr::New(ctx.GetAstPool(), std::move(*left), Pos(peek));
      be->right.emplace_back(*op, std::move(*right));
      left = be;
    }
    return left;
  }

  // multiplicative_expr = power_expr { mul_op power_expr }
  absl::StatusOr<BinaryExprPtr> parse_multiplicative_expr() {
    return parse_binary_left(
        [this]() { return parse_power_expr(); },
        [](TokenType t) { return MultiplicativeOpFromToken(t); });
  }

  // additive_expr = multiplicative_expr { add_op multiplicative_expr }
  absl::StatusOr<BinaryExprPtr> parse_additive_expr() {
    return parse_binary_left(
        [this]() { return parse_multiplicative_expr(); },
        [](TokenType t) { return AdditiveOpFromToken(t); });
  }

  // cmp_expr = additive_expr { cmp_op additive_expr }
  absl::StatusOr<BinaryExprPtr> parse_cmp_expr() {
    return parse_binary_left(
        [this]() { return parse_additive_expr(); },
        [](TokenType t) { return CmpOpFromToken(t); });
  }

  // logic_expr = cmp_expr { logic_op cmp_expr }
  absl::StatusOr<BinaryExprPtr> parse_logic_expr() {
    return parse_binary_left(
        [this]() { return parse_cmp_expr(); },
        [](TokenType t) { return LogicOpFromToken(t); });
  }

  // assign = logic_expr [ '?' expression ':' expression ] [ assign_op expression ]
  absl::StatusOr<BinaryExprPtr> parse_assign() {
    // Parse the base expression (logic_expr)
    auto left = parse_logic_expr();
    if (!left.ok()) return left.status();

    // Check for ternary: logic_expr '?' expression ':' expression
    if (lex.PeekIs(TOKEN_QUESTION)) {
      lex.Next();  // consume '?'
      auto true_expr = parse_expression();
      if (!true_expr.ok()) return true_expr.status();

      Token colon = lex.Expect(TOKEN_COLON);
      if (colon.Is(TOKEN_ERROR)) {
        return absl::InvalidArgumentError("expected ':' in ternary expression");
      }

      auto false_expr = parse_expression();
      if (!false_expr.ok()) return false_expr.status();

      auto* se = ctx.GetAstPool().New<SelectExpr>();
      se->cond = std::move(*left);
      se->true_false_operands = std::make_tuple(Operand(std::move(*true_expr)), Operand(std::move(*false_expr)));
      se->position = 0;

      // Wrap ternary in binary
      auto be = BinaryExpr::New(ctx.GetAstPool(), SelectExprPtr(se), 0);

      // Check for assign after ternary
      Token peek = lex.Peek();
      auto assign_op = AssignOpFromToken(peek.type);
      if (assign_op.has_value()) {
        lex.Next();
        auto rhs = parse_expression();
        if (!rhs.ok()) return rhs.status();
        be->right.emplace_back(*assign_op, *rhs);
      }
      return be;
    }

    // Check for assign: left assign_op expression
    Token peek = lex.Peek();
    auto assign_op = AssignOpFromToken(peek.type);
    if (assign_op.has_value()) {
      lex.Next();
      auto rhs = parse_expression();
      if (!rhs.ok()) return rhs.status();

      auto be = BinaryExpr::New(ctx.GetAstPool(), std::move(*left), Pos(peek));
      be->right.emplace_back(*assign_op, std::move(*rhs));
      return be;
    }

    return left;
  }

  // expression = assign
  absl::StatusOr<BinaryExprPtr> parse_expression() {
    return parse_assign();
  }

  // ---- Statement parsing ----

  // return_statement = 'return' [ expression ] ';'
  absl::StatusOr<ReturnStatement> parse_return_statement() {
    lex.Next();  // consume 'return'

    ReturnStatement rs;
    if (!lex.PeekIs(TOKEN_SEMICOLON)) {
      auto expr = parse_expression();
      if (!expr.ok()) return expr.status();
      rs.expr = *expr;
    }

    Token semi = lex.Expect(TOKEN_SEMICOLON);
    if (semi.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ';' after return statement");
    }
    return rs;
  }

  // expr_statement = expression ';'
  absl::StatusOr<ExpressionStatement> parse_expr_statement() {
    auto expr = parse_expression();
    if (!expr.ok()) return expr.status();

    Token semi = lex.Expect(TOKEN_SEMICOLON);
    if (semi.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ';' after expression statement");
    }

    ExpressionStatement es;
    es.expr = *expr;
    return es;
  }

  // choice_statement = '(' expression ')' '{' statements '}'
  absl::StatusOr<ChoiceStatement> parse_choice_statement() {
    Token lparen = lex.Expect(TOKEN_LPAREN);
    if (lparen.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '('");
    }

    auto expr = parse_expression();
    if (!expr.ok()) return expr.status();

    Token rparen = lex.Expect(TOKEN_RPAREN);
    if (rparen.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ')'");
    }

    Token lbrace = lex.Expect(TOKEN_LBRACE);
    if (lbrace.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '{'");
    }

    auto stmts = parse_statements();
    if (!stmts.ok()) return stmts.status();

    Token rbrace = lex.Expect(TOKEN_RBRACE);
    if (rbrace.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '}'");
    }

    ChoiceStatement cs;
    cs.expr = *expr;
    cs.statements = std::move(*stmts);
    return cs;
  }

  // while_statement = 'while' choice_statement
  absl::StatusOr<WhileStatement> parse_while_statement() {
    lex.Next();  // consume 'while'

    auto cs = parse_choice_statement();
    if (!cs.ok()) return cs.status();

    WhileStatement ws;
    ws.body = *cs;
    return ws;
  }

  // ifelse_statement = 'if' choice_statement { 'elif' choice_statement } [ 'else' '{' statements '}' ]
  absl::StatusOr<IfElseStatement> parse_ifelse_statement() {
    lex.Next();  // consume 'if'

    auto if_cs = parse_choice_statement();
    if (!if_cs.ok()) return if_cs.status();

    IfElseStatement ies;
    ies.if_statement = *if_cs;

    // elif branches
    while (lex.PeekIs(TOKEN_ELIF)) {
      lex.Next();  // consume 'elif'
      auto elif_cs = parse_choice_statement();
      if (!elif_cs.ok()) return elif_cs.status();
      ies.elif_statements.push_back(*elif_cs);
    }

    // else branch
    if (lex.PeekIs(TOKEN_ELSE)) {
      lex.Next();  // consume 'else'
      Token lbrace = lex.Expect(TOKEN_LBRACE);
      if (lbrace.Is(TOKEN_ERROR)) {
        return absl::InvalidArgumentError("expected '{' after 'else'");
      }

      auto else_stmts = parse_statements();
      if (!else_stmts.ok()) return else_stmts.status();

      Token rbrace = lex.Expect(TOKEN_RBRACE);
      if (rbrace.Is(TOKEN_ERROR)) {
        return absl::InvalidArgumentError("expected '}'");
      }
      ies.else_statements = std::move(*else_stmts);
    }

    return ies;
  }

  // statements = { statement }
  absl::StatusOr<std::vector<Statement>> parse_statements() {
    std::vector<Statement> result;

    while (true) {
      Token peek = lex.Peek();

      if (peek.Is(TOKEN_CONTINUE)) {
        lex.Next();
        Token semi = lex.Expect(TOKEN_SEMICOLON);
        if (semi.Is(TOKEN_ERROR)) {
          break;
        }
        ContinueStatement cs;
        cs.position = Pos(peek);
        result.push_back(cs);
      } else if (peek.Is(TOKEN_BREAK)) {
        lex.Next();
        Token semi = lex.Expect(TOKEN_SEMICOLON);
        if (semi.Is(TOKEN_ERROR)) {
          break;
        }
        BreakStatement bs;
        bs.position = Pos(peek);
        result.push_back(bs);
      } else if (peek.Is(TOKEN_RETURN)) {
        auto rs = parse_return_statement();
        if (!rs.ok()) return rs.status();
        result.push_back(*rs);
      } else if (peek.Is(TOKEN_WHILE)) {
        auto ws = parse_while_statement();
        if (!ws.ok()) return ws.status();
        result.push_back(*ws);
      } else if (peek.Is(TOKEN_IF)) {
        auto ies = parse_ifelse_statement();
        if (!ies.ok()) return ies.status();
        result.push_back(*ies);
      } else if (peek.Is(TOKEN_IDENTIFIER) || peek.Is(TOKEN_NUMBER) || peek.Is(TOKEN_STRING) ||
                 peek.Is(TOKEN_BOOL) || peek.Is(TOKEN_AUTO) || peek.Is(TOKEN_LPAREN) ||
                 peek.Is(TOKEN_LBRACKET) || peek.Is(TOKEN_MINUS) || peek.Is(TOKEN_NOT)) {
        auto es = parse_expr_statement();
        if (!es.ok()) return es.status();
        result.push_back(*es);
      } else {
        break;
      }
    }
    return result;
  }

  // block = '{' statements '}'
  absl::StatusOr<Block> parse_block() {
    Token lbrace = lex.Expect(TOKEN_LBRACE);
    if (lbrace.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '{'");
    }

    auto stmts = parse_statements();
    if (!stmts.ok()) return stmts.status();

    Token rbrace = lex.Expect(TOKEN_RBRACE);
    if (rbrace.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '}'");
    }

    Block block;
    block.statements = std::move(*stmts);
    return block;
  }

  // Read a full type name including template args, e.g. "simd_vector<f32>"
  absl::StatusOr<std::string> read_type_name() {
    Token dtype_tok = lex.Next();
    std::string name(dtype_tok.text);
    // Handle template types like simd_vector<f32>, dyn_obj<schema_name>, table<schema_name>
    if (lex.PeekIs(TOKEN_LT)) {
      lex.Next();  // consume '<'
      name.push_back('<');
      // Read inner type args (identifiers, commas, colons, etc. until '>')
      while (!lex.PeekIs(TOKEN_GT) && !lex.PeekIs(TOKEN_EOF)) {
        Token t = lex.Next();
        if (t.Is(TOKEN_ERROR)) {
          return absl::InvalidArgumentError(fmt::format("unexpected token in template type: {}", name));
        }
        name.append(t.text);
        if (lex.PeekIs(TOKEN_COMMA)) {
          lex.Next();
          name.push_back(',');
        }
      }
      if (lex.PeekIs(TOKEN_GT)) {
        lex.Next();  // consume '>'
        name.push_back('>');
      } else {
        return absl::InvalidArgumentError(
            fmt::format("expected '>' to close template type, got end of input in: {}", name));
      }
    }
    return name;
  }

  // func_arg = dtype identifier
  absl::StatusOr<FunctionArg> parse_func_arg() {
    auto type_name_result = read_type_name();
    if (!type_name_result.ok()) return type_name_result.status();
    std::string type_name = *type_name_result;

    auto dtype_found = Symbols::FindDType(type_name);
    if (!dtype_found.has_value()) {
      return absl::InvalidArgumentError(fmt::format("unknown type: {}", type_name));
    }

    Token name_tok = lex.Expect(TOKEN_IDENTIFIER);
    if (name_tok.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected argument name");
    }
    if (!IsValidIdentifier(name_tok.text)) {
      return absl::InvalidArgumentError(
          fmt::format("invalid argument name: '{}' is a reserved name", name_tok.text));
    }

    FunctionArg fa;
    fa.dtype = dtype_found->first;
    fa.attr = dtype_found->second;
    fa.name = std::string(name_tok.text);
    return fa;
  }

  // func = dtype identifier '(' [ func_arg { ',' func_arg } ] ')' block
  absl::StatusOr<Function> parse_func() {
    uint32_t func_pos = static_cast<uint32_t>(lex.Position());
    auto ret_type_name_result = read_type_name();
    if (!ret_type_name_result.ok()) return ret_type_name_result.status();
    std::string ret_type_name = *ret_type_name_result;

    auto ret_type = Symbols::FindDType(ret_type_name);
    if (!ret_type.has_value()) {
      return absl::InvalidArgumentError(fmt::format("unknown return type: {}", ret_type_name));
    }

    Token name_tok = lex.Expect(TOKEN_IDENTIFIER);
    if (name_tok.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected function name");
    }
    if (!IsValidIdentifier(name_tok.text)) {
      return absl::InvalidArgumentError(
          fmt::format("invalid function name: '{}' is a reserved name", name_tok.text));
    }

    Token lparen = lex.Expect(TOKEN_LPAREN);
    if (lparen.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected '('");
    }

    std::optional<std::vector<FunctionArg>> args;
    if (!lex.PeekIs(TOKEN_RPAREN)) {
      std::vector<FunctionArg> arg_list;
      auto arg = parse_func_arg();
      if (!arg.ok()) return arg.status();
      arg_list.push_back(*arg);

      while (lex.ConsumeIf(TOKEN_COMMA)) {
        arg = parse_func_arg();
        if (!arg.ok()) return arg.status();
        arg_list.push_back(*arg);
      }
      args = std::move(arg_list);
    }

    Token rparen = lex.Expect(TOKEN_RPAREN);
    if (rparen.Is(TOKEN_ERROR)) {
      return absl::InvalidArgumentError("expected ')'");
    }

    auto body = parse_block();
    if (!body.ok()) return body.status();

    Function func;
    func.return_type = ret_type->first;
    func.name = std::string(name_tok.text);
    func.args = std::move(args);
    func.body = *body;
    func.position = func_pos;
    return func;
  }
};

// ---- Entry points ----

absl::StatusOr<Function> parse_function_ast(ParseContext& ctx, const std::string& source) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source);

  Lexer lex(source);
  Parser parser{lex, ctx};

  auto result = parser.parse_func();

  auto parse_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time);
  start_time = std::chrono::high_resolution_clock::now();
  ctx.SetParseCost(parse_duration);

  if (!result.ok()) {
    std::string err_detail;
    if (lex.HasError()) {
      err_detail = lex.GetError();
    } else {
      err_detail = std::string(result.status().message());
    }
    return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, err_detail));
  }

  ctx.SetFuncDesc(result->ToFuncDesc());
  auto rc = result->Validate(ctx);
  if (!rc.ok()) {
    return rc;
  }
  ctx.SetParseValidateCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));

  return *result;
}

absl::StatusOr<std::vector<Function>> parse_functions_ast(ParseContext& ctx, const std::string& source) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source);

  Lexer lex(source);
  Parser parser{lex, ctx};

  std::vector<Function> funcs;
  while (!lex.IsEOF()) {
    Token peek = lex.Peek();
    if (peek.Is(TOKEN_EOF)) break;

    auto func = parser.parse_func();
    if (!func.ok()) {
      std::string err_detail;
      if (lex.HasError()) {
        err_detail = lex.GetError();
      } else {
        err_detail = std::string(func.status().message());
      }
      return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, err_detail));
    }
    funcs.push_back(*func);
  }

  ctx.SetParseCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  start_time = std::chrono::high_resolution_clock::now();

  std::set<std::string> func_names;
  for (size_t i = 0; i < funcs.size(); i++) {
    ctx.SetFunctionCursor(i);
    ctx.SetFuncDesc(funcs[i].ToFuncDesc(), i);
    auto rc = funcs[i].Validate(ctx);
    if (!rc.ok()) {
      return rc;
    }
    if (!func_names.insert(funcs[i].name).second) {
      return absl::InvalidArgumentError(fmt::format("duplicate function name:{}", funcs[i].name));
    }
  }

  ctx.SetParseValidateCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));

  return funcs;
}

absl::StatusOr<Expression> parse_expression_ast(ParseContext& ctx, const std::string& source,
                                                const FunctionDesc& desc) {
  auto start_time = std::chrono::high_resolution_clock::now();
  ctx.SetSource(source, false);

  Lexer lex(source);
  Parser parser{lex, ctx};

  auto result = parser.parse_expression();

  ctx.SetParseCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));
  start_time = std::chrono::high_resolution_clock::now();

  if (!result.ok()) {
    std::string err_detail;
    if (lex.HasError()) {
      err_detail = lex.GetError();
    } else {
      err_detail = std::string(result.status().message());
    }
    return absl::InvalidArgumentError(fmt::format("parse {} failed with ast_error:{}", source, err_detail));
  }

  Expression expr;
  ctx.SetFuncDesc(desc);
  expr.expr = *result;
  auto rc = (*result)->Validate(ctx, expr.rpn_expr);
  if (!rc.ok()) {
    return rc.status();
  }
  ctx.SetParseValidateCost(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time));

  return expr;
}

}  // namespace ast
}  // namespace rapidudf
