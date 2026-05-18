#include <chrono>
#include <iostream>
#include <string>

#include "rapidudf/ast/grammar.h"
#include "rapidudf/ast/context.h"
#include "rapidudf/ast/lexer.h"
#include "rapidudf/meta/dtype.h"

using namespace rapidudf;
using namespace rapidudf::ast;

constexpr const char* kTestExpressions[] = {
    "(y + x)",
    "2 * (y + x)",
    "(2 * y + 2 * x)",
    "((1.23 * x^2) / y) - 123.123",
    "(y + x / y) * (x - y / x)",
    "x / ((x + y) + (x - y)) / y",
    "1 - ((x * y) + (y / x)) - 3",
    "(5.5 + x) + (2 * x - 2 / 3 * y) * (x / 3 + y / 4) + (y + 7.7)",
    "1.1*x^1 + 2.2*y^2 - 3.3*x^3 + 4.4*y^15 - 5.5*x^23 + 6.6*y^55",
    "2 * x + 3 * y",
    "1 - 2 * x + 3 * y",
    "111.111 - 2 * x + 3 * y / 333.333",
    "(x^2 / (2 * 3 / y)) - x / 2",
    "x + (3 * y - 2 / x * 3) - y",
    "((y + (x * 2.2)) <= (x + y + 1.1))?(x - y):(x * y) + 2 * 3 / x",
    "x + y * 2 - 3 / x + 4 * y - 5 * x + 6 * y",
    "((y + (x * 2.2)) <= (x + y + 1.1))?(x - y):(x * y) + 2 * 3 / x",
};
constexpr size_t kNumExpressions = sizeof(kTestExpressions) / sizeof(kTestExpressions[0]);

int main() {
  constexpr int kRounds = 1000;

  // Warm up
  for (size_t i = 0; i < kNumExpressions; ++i) {
    ParseContext warmup_ctx;
    warmup_ctx.AddLocalVar("x", DType(DATA_F64), nullptr);
    warmup_ctx.AddLocalVar("y", DType(DATA_F64), nullptr);
    parse_expression_ast(warmup_ctx, kTestExpressions[i], FunctionDesc{});
  }

  // Benchmark: Full parse (including ParseContext creation + validation)
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < kRounds; ++round) {
      for (size_t i = 0; i < kNumExpressions; ++i) {
        ParseContext expr_ctx;
        expr_ctx.AddLocalVar("x", DType(DATA_F64), nullptr);
        expr_ctx.AddLocalVar("y", DType(DATA_F64), nullptr);
        auto result = parse_expression_ast(expr_ctx, kTestExpressions[i], FunctionDesc{});
        if (!result.ok()) {
          std::cerr << "Parse failed: " << result.status().message() << std::endl;
          return 1;
        }
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);
    size_t total = kRounds * kNumExpressions;
    std::cout << "[Full] Parsed " << total << " expressions in " << elapsed.count() / 1000 << " ms" << std::endl;
    std::cout << "[Full] Rate: " << static_cast<int>(static_cast<double>(total) / (elapsed.count() / 1e6))
              << " parses/sec, Avg: " << static_cast<double>(elapsed.count()) / total << " us/parse" << std::endl;
  }

  // Benchmark 2: Lexer only (tokenize without parsing)
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < kRounds; ++round) {
      for (size_t i = 0; i < kNumExpressions; ++i) {
        Lexer lex(kTestExpressions[i]);
        while (!lex.IsEOF()) {
          Token t = lex.Next();
          if (t.type == TOKEN_ERROR) break;
        }
      }
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);
    size_t total = kRounds * kNumExpressions;
    std::cout << "[Lexer] Tokenized " << total << " expressions in " << elapsed.count() / 1000 << " ms" << std::endl;
    std::cout << "[Lexer] Rate: " << static_cast<int>(static_cast<double>(total) / (elapsed.count() / 1e6))
              << " tokenizes/sec, Avg: " << static_cast<double>(elapsed.count()) / total << " us/tokenize" << std::endl;
  }

  return 0;
}
