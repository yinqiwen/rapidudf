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

#include "rapidudf/ast/lexer.h"

#include <unordered_map>

namespace rapidudf {
namespace ast {

static bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool IsDigit(char c) { return c >= '0' && c <= '9'; }
static bool IsAlphaNum(char c) { return IsAlpha(c) || IsDigit(c); }

static const std::unordered_map<std::string_view, TokenType> kKeywords = {
    {"if", TOKEN_IF},       {"elif", TOKEN_ELIF},     {"else", TOKEN_ELSE},   {"while", TOKEN_WHILE},
    {"return", TOKEN_RETURN}, {"auto", TOKEN_AUTO},   {"break", TOKEN_BREAK}, {"continue", TOKEN_CONTINUE},
    {"true", TOKEN_BOOL},   {"false", TOKEN_BOOL},
};

Lexer::Lexer(std::string_view source) : source_(source) {}

bool Lexer::IsEOF() const { return pos_ >= source_.size(); }

void Lexer::SkipWhitespaceAndComments() {
  while (pos_ < source_.size()) {
    char c = source_[pos_];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      ++pos_;
    } else if (c == '/' && pos_ + 1 < source_.size() && source_[pos_ + 1] == '/') {
      // Line comment: skip until end of line
      pos_ += 2;
      while (pos_ < source_.size() && source_[pos_] != '\n') {
        ++pos_;
      }
    } else {
      break;
    }
  }
}

Token Lexer::ReadNumber() {
  size_t start = pos_;
  bool has_dot = false;

  while (pos_ < source_.size()) {
    char c = source_[pos_];
    if (IsDigit(c)) {
      ++pos_;
    } else if (c == '.' && !has_dot && pos_ + 1 < source_.size() && IsDigit(source_[pos_ + 1])) {
      has_dot = true;
      ++pos_;
    } else {
      break;
    }
  }

  // Check for type suffix like _f32, _i32
  if (pos_ < source_.size() && source_[pos_] == '_') {
    size_t suffix_start = pos_;
    ++pos_;
    while (pos_ < source_.size() && IsAlphaNum(source_[pos_])) {
      ++pos_;
    }
    // Validate suffix is a known type suffix
    // We'll let the parser validate the suffix; just consume it as part of the number token
    return Token{TOKEN_NUMBER, source_.substr(start, pos_ - start), start};
  }

  return Token{TOKEN_NUMBER, source_.substr(start, pos_ - start), start};
}

Token Lexer::ReadString() {
  size_t start = pos_;
  ++pos_;  // skip opening quote

  while (pos_ < source_.size() && source_[pos_] != '"') {
    if (source_[pos_] == '\\') {
      ++pos_;  // skip escaped char
    }
    if (pos_ < source_.size()) {
      ++pos_;
    }
  }

  if (pos_ < source_.size()) {
    ++pos_;  // skip closing quote
  } else {
    SetError("unterminated string literal");
    return Token{TOKEN_ERROR, source_.substr(start, pos_ - start), start};
  }

  return Token{TOKEN_STRING, source_.substr(start, pos_ - start), start};
}

Token Lexer::ReadIdentifierOrKeyword() {
  size_t start = pos_;

  // First char: letter or underscore
  ++pos_;

  // Following chars: letter, digit, underscore, or colon
  while (pos_ < source_.size() && (IsAlphaNum(source_[pos_]) || source_[pos_] == ':')) {
    ++pos_;
  }

  std::string_view text = source_.substr(start, pos_ - start);

  auto it = kKeywords.find(text);
  if (it != kKeywords.end()) {
    return Token{it->second, text, start};
  }

  return Token{TOKEN_IDENTIFIER, text, start};
}

Token Lexer::Next() {
  if (peeked_.has_value()) {
    Token t = *peeked_;
    peeked_.reset();
    return t;
  }

  SkipWhitespaceAndComments();

  if (pos_ >= source_.size()) {
    return Token{TOKEN_EOF, {}, pos_};
  }

  size_t start = pos_;
  char c = source_[pos_];

  // Numbers
  if (IsDigit(c) || (c == '.' && pos_ + 1 < source_.size() && IsDigit(source_[pos_ + 1]))) {
    return ReadNumber();
  }

  // Identifiers and keywords
  if (IsAlpha(c)) {
    return ReadIdentifierOrKeyword();
  }

  // String literals
  if (c == '"') {
    return ReadString();
  }

  // Operators and separators
  ++pos_;
  switch (c) {
    case '+':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_PLUS_ASSIGN, source_.substr(start, 2), start};
      }
      return Token{TOKEN_PLUS, source_.substr(start, 1), start};
    case '-':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_MINUS_ASSIGN, source_.substr(start, 2), start};
      }
      return Token{TOKEN_MINUS, source_.substr(start, 1), start};
    case '*':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_STAR_ASSIGN, source_.substr(start, 2), start};
      }
      return Token{TOKEN_STAR, source_.substr(start, 1), start};
    case '/':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_SLASH_ASSIGN, source_.substr(start, 2), start};
      }
      return Token{TOKEN_SLASH, source_.substr(start, 1), start};
    case '%':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_PERCENT_ASSIGN, source_.substr(start, 2), start};
      }
      return Token{TOKEN_PERCENT, source_.substr(start, 1), start};
    case '^':
      return Token{TOKEN_CARET, source_.substr(start, 1), start};
    case '=':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_EQ, source_.substr(start, 2), start};
      }
      return Token{TOKEN_ASSIGN, source_.substr(start, 1), start};
    case '!':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_NE, source_.substr(start, 2), start};
      }
      return Token{TOKEN_NOT, source_.substr(start, 1), start};
    case '<':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_LE, source_.substr(start, 2), start};
      }
      return Token{TOKEN_LT, source_.substr(start, 1), start};
    case '>':
      if (pos_ < source_.size() && source_[pos_] == '=') {
        ++pos_;
        return Token{TOKEN_GE, source_.substr(start, 2), start};
      }
      return Token{TOKEN_GT, source_.substr(start, 1), start};
    case '&':
      if (pos_ < source_.size() && source_[pos_] == '&') {
        ++pos_;
        return Token{TOKEN_AND, source_.substr(start, 2), start};
      }
      SetError("unexpected character '&'");
      return Token{TOKEN_ERROR, source_.substr(start, 1), start};
    case '|':
      if (pos_ < source_.size() && source_[pos_] == '|') {
        ++pos_;
        return Token{TOKEN_OR, source_.substr(start, 2), start};
      }
      SetError("unexpected character '|'");
      return Token{TOKEN_ERROR, source_.substr(start, 1), start};
    case '?':
      return Token{TOKEN_QUESTION, source_.substr(start, 1), start};
    case ':':
      return Token{TOKEN_COLON, source_.substr(start, 1), start};
    case '(':
      return Token{TOKEN_LPAREN, source_.substr(start, 1), start};
    case ')':
      return Token{TOKEN_RPAREN, source_.substr(start, 1), start};
    case '{':
      return Token{TOKEN_LBRACE, source_.substr(start, 1), start};
    case '}':
      return Token{TOKEN_RBRACE, source_.substr(start, 1), start};
    case '[':
      return Token{TOKEN_LBRACKET, source_.substr(start, 1), start};
    case ']':
      return Token{TOKEN_RBRACKET, source_.substr(start, 1), start};
    case ',':
      return Token{TOKEN_COMMA, source_.substr(start, 1), start};
    case ';':
      return Token{TOKEN_SEMICOLON, source_.substr(start, 1), start};
    case '.':
      return Token{TOKEN_DOT, source_.substr(start, 1), start};
    default:
      SetError(std::string("unexpected character '") + c + "'");
      return Token{TOKEN_ERROR, source_.substr(start, 1), start};
  }
}

Token Lexer::Peek() {
  if (!peeked_.has_value()) {
    peeked_ = Next();
  }
  return *peeked_;
}

bool Lexer::PeekIs(TokenType type) { return Peek().type == type; }

Token Lexer::Expect(TokenType type) {
  Token t = Next();
  if (t.type != type) {
    SetError(std::string("expected token type ") + std::to_string(type) + " but got " + std::to_string(t.type));
    return Token{TOKEN_ERROR, t.text, t.position};
  }
  return t;
}

std::optional<Token> Lexer::ConsumeIf(TokenType type) {
  if (PeekIs(type)) {
    return Next();
  }
  return std::nullopt;
}

void Lexer::SetError(const std::string& msg) { error_ = msg; }

}  // namespace ast
}  // namespace rapidudf
