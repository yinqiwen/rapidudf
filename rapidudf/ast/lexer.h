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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rapidudf {
namespace ast {

enum TokenType {
  // Literals
  TOKEN_IDENTIFIER,
  TOKEN_NUMBER,
  TOKEN_STRING,
  TOKEN_BOOL,

  // Keywords
  TOKEN_IF,
  TOKEN_ELIF,
  TOKEN_ELSE,
  TOKEN_WHILE,
  TOKEN_RETURN,
  TOKEN_AUTO,
  TOKEN_BREAK,
  TOKEN_CONTINUE,

  // Operators
  TOKEN_PLUS,           // +
  TOKEN_MINUS,          // -
  TOKEN_STAR,           // *
  TOKEN_SLASH,          // /
  TOKEN_PERCENT,        // %
  TOKEN_CARET,          // ^
  TOKEN_ASSIGN,         // =
  TOKEN_PLUS_ASSIGN,    // +=
  TOKEN_MINUS_ASSIGN,   // -=
  TOKEN_STAR_ASSIGN,    // *=
  TOKEN_SLASH_ASSIGN,   // /=
  TOKEN_PERCENT_ASSIGN, // %=
  TOKEN_EQ,             // ==
  TOKEN_NE,             // !=
  TOKEN_LT,             // <
  TOKEN_LE,             // <=
  TOKEN_GT,             // >
  TOKEN_GE,             // >=
  TOKEN_AND,            // &&
  TOKEN_OR,             // ||
  TOKEN_NOT,            // !
  TOKEN_QUESTION,       // ?
  TOKEN_COLON,          // :

  // Separators
  TOKEN_LPAREN,    // (
  TOKEN_RPAREN,    // )
  TOKEN_LBRACE,    // {
  TOKEN_RBRACE,    // }
  TOKEN_LBRACKET,  // [
  TOKEN_RBRACKET,  // ]
  TOKEN_COMMA,     // ,
  TOKEN_SEMICOLON, // ;
  TOKEN_DOT,       // .

  // Special
  TOKEN_EOF,
  TOKEN_ERROR,
};

struct Token {
  TokenType type;
  std::string_view text;
  size_t position;  // offset from source start

  bool Is(TokenType t) const { return type == t; }
  bool IsNot(TokenType t) const { return type != t; }
};

class Lexer {
 public:
  explicit Lexer(std::string_view source);

  // Get next token, skipping whitespace and comments
  Token Next();

  // Peek at the next token without consuming it
  Token Peek();

  // Check if next token matches the given type
  bool PeekIs(TokenType type);

  // Consume next token, assert it matches the given type, return it
  Token Expect(TokenType type);

  // Try to consume a token of the given type; returns nullopt if not matching
  std::optional<Token> ConsumeIf(TokenType type);

  // Current position in source
  size_t Position() const { return pos_; }

  // Whether we've reached the end
  bool IsEOF() const;

  // Get the source string
  std::string_view Source() const { return source_; }

  // Set error message
  void SetError(const std::string& msg);
  const std::string& GetError() const { return error_; }
  bool HasError() const { return !error_.empty(); }

 private:
  void SkipWhitespaceAndComments();
  Token ReadNumber();
  Token ReadString();
  Token ReadIdentifierOrKeyword();

  std::string_view source_;
  size_t pos_ = 0;
  std::optional<Token> peeked_;
  std::string error_;
};

}  // namespace ast
}  // namespace rapidudf
