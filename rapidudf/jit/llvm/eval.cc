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
#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "rapidudf/ast/expression.h"
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/jit_session.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, DType dtype,
                                              const std::deque<RPNEvalNode>& nodes) {
  std::vector<ValuePtr> operands;

  for (auto& node : nodes) {
    const OpToken* op_ptr = std::get_if<OpToken>(&node);
    if (op_ptr != nullptr) {
      auto op = *op_ptr;
      int operand_count = get_operand_count(op);
      if (operand_count <= 0) {
        RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("get_operand_count failed for:{}", op)));
      }
      if (operands.size() < static_cast<size_t>(operand_count)) {
        RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format(
            "Only {} operands in stack for op:{}, which required {} args.", operands.size(), op, operand_count)));
      }
      absl::StatusOr<ValuePtr> result;
      if (operand_count == 1) {
        result = BuildIR(ctx, op, operands[operands.size() - 1]);
        operands.pop_back();
      } else if (operand_count == 2) {
        result = BuildIR(ctx, op, operands[operands.size() - 2], operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
      } else if (operand_count == 3) {
        result = BuildIR(ctx, op, operands[operands.size() - 3], operands[operands.size() - 2],
                         operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
        operands.pop_back();
      } else {
        RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("Unsupported op:{}", op)));
      }
      if (!result.ok()) {
        return result.status();
      }
      operands.emplace_back(result.value());
    } else {
      operands.emplace_back(std::get<ValuePtr>(node));
    }
  }
  if (operands.size() != 1) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("After eval expr, {} operands rest.", operands.size())));
  }
  return operands[0];
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(FunctionCompileContextPtr ctx, const ast::RPN& rpn) {
  if (rpn.dtype.IsInvalid()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("Invalid rpn dtype to eval")));
  }
  std::deque<RPNEvalNode> eval_nodes;
  bool is_simd_vector_expr = false;
  bool is_simd_column_expr = false;
  for (auto& node : rpn.nodes) {
    auto result = std::visit(
        [&](auto&& arg) -> absl::Status {
          using T = std::decay_t<decltype(arg)>;
          absl::StatusOr<ValuePtr> val_result;
          if constexpr (std::is_same_v<T, bool>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, ast::VarDefine>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, ast::VarAccessor>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, ast::ConstantNumber>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, ast::Array>) {
            val_result = BuildIR(ctx, arg);
          } else if constexpr (std::is_same_v<T, ast::SelectRPNNodePtr>) {
            val_result = BuildIR(ctx, arg);
          } else {
            OpToken op = arg;
            eval_nodes.emplace_back(op);
            return absl::OkStatus();
          }
          if (!val_result.ok()) {
            return val_result.status();
          }
          auto operand = val_result.value();
          if (operand->GetDType().IsSimdVector()) {
            is_simd_vector_expr = true;
          } else if (operand->GetDType().IsSimdColumnPtr()) {
            is_simd_column_expr = true;
          }
          eval_nodes.emplace_back(operand);
          return absl::OkStatus();
        },
        node);
    if (!result.ok()) {
      return result;
    }
  }

  if (is_simd_vector_expr && is_simd_column_expr) {
    RUDF_LOG_ERROR_STATUS(
        absl::InvalidArgumentError(fmt::format("Vector expression can NOT have both vector&column data.")));
  }
  if (eval_nodes.size() > 1 && (is_simd_vector_expr || is_simd_column_expr)) {
    if (opts_.fuse_vector_ops) {
      return BuildVectorEvalIR2(ctx, rpn.dtype, eval_nodes);
    }
  }
  return BuildIR(ctx, rpn.dtype, eval_nodes);
}

absl::StatusOr<ValuePtr> JitCompiler::BuildVectorEvalIR(FunctionCompileContextPtr ctx, DType result_dtype,
                                                        std::deque<RPNEvalNode>& nodes) {
  const OpToken* last_op = std::get_if<OpToken>(&nodes[nodes.size() - 1]);
  std::optional<ValuePtr> assign_to;
  switch (*last_op) {
    case OP_ASSIGN: {
      assign_to = std::get<ValuePtr>(nodes.front());
      nodes.pop_front();
      nodes.pop_back();
      break;
    }
    case OP_PLUS_ASSIGN: {
      nodes[nodes.size() - 1] = OP_PLUS;
      assign_to = std::get<ValuePtr>(nodes.front());
      break;
    }
    case OP_MOD_ASSIGN: {
      nodes[nodes.size() - 1] = OP_MOD;
      assign_to = std::get<ValuePtr>(nodes.front());
      break;
    }
    case OP_MINUS_ASSIGN: {
      nodes[nodes.size() - 1] = OP_MINUS;
      assign_to = std::get<ValuePtr>(nodes.front());
      break;
    }
    case OP_MULTIPLY_ASSIGN: {
      nodes[nodes.size() - 1] = OP_MULTIPLY;
      assign_to = std::get<ValuePtr>(nodes.front());
      break;
    }
    case OP_DIVIDE_ASSIGN: {
      nodes[nodes.size() - 1] = OP_DIVIDE;
      assign_to = std::get<ValuePtr>(nodes.front());
      break;
    }
    default: {
      break;
    }
  }

  auto* ir_builder = GetSession()->GetIRBuilder();
  ::llvm::Type* rpn_value_type = ::llvm::StructType::getTypeByName(ir_builder->getContext(), "rpn_value");
  ::llvm::Type* vector_type = ::llvm::StructType::getTypeByName(ir_builder->getContext(), "simd_vector");
  auto* stack_val = ir_builder->CreateAlloca(rpn_value_type, ir_builder->getInt64(nodes.size()));

  for (size_t i = 0; i < nodes.size(); i++) {
    auto element_ptr =
        ir_builder->CreateInBoundsGEP(rpn_value_type, stack_val, std::vector<::llvm::Value*>{ir_builder->getInt64(i)});
    auto dtype_field_ptr = ir_builder->CreateInBoundsGEP(
        rpn_value_type, element_ptr, std::vector<::llvm::Value*>{ir_builder->getInt32(0), ir_builder->getInt32(0)});
    auto val_field_ptr = ir_builder->CreateInBoundsGEP(
        rpn_value_type, element_ptr, std::vector<::llvm::Value*>{ir_builder->getInt32(0), ir_builder->getInt32(1)});
    auto size_field_ptr = ir_builder->CreateInBoundsGEP(
        vector_type, val_field_ptr, std::vector<::llvm::Value*>{ir_builder->getInt32(0), ir_builder->getInt32(0)});
    auto ptr_field_ptr = ir_builder->CreateInBoundsGEP(
        vector_type, val_field_ptr, std::vector<::llvm::Value*>{ir_builder->getInt32(0), ir_builder->getInt32(1)});

    auto& node = nodes[i];
    const OpToken* op_ptr = std::get_if<OpToken>(&node);
    if (op_ptr == nullptr) {
      auto value = std::get<ValuePtr>(node);
      ir_builder->CreateStore(ir_builder->getInt64(value->GetDType().Control()), dtype_field_ptr);
      if (value->GetDType().IsSimdVector() || value->GetDType().IsStringView()) {
        ir_builder->CreateStore(value->GetValue(), val_field_ptr);
      } else {
        ir_builder->CreateStore(value->GetValue(), size_field_ptr);
        // ir_builder->CreateStore(value->GetValue(), val_field_ptr);
      }
    } else {
      ir_builder->CreateStore(ir_builder->getInt64(0), dtype_field_ptr);
      ir_builder->CreateStore(ir_builder->getInt64(static_cast<int64_t>(*op_ptr)), size_field_ptr);
      // ir_builder->CreateStore(ir_builder->getInt64(static_cast<int64_t>(*op_ptr)), val_field_ptr);
    }
  }
  auto* span_type = ::llvm::StructType::getTypeByName(ir_builder->getContext(), "absl_span");
  auto* span_val = ir_builder->CreateAlloca(span_type);
  auto size_val = ir_builder->getInt64(nodes.size());
  ::llvm::Value* zero = ir_builder->getInt32(0);
  ::llvm::Value* offset = ir_builder->getInt32(1);
  auto size_field_ptr = ir_builder->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  ir_builder->CreateStore(size_val, size_field_ptr);
  offset = ir_builder->getInt32(0);
  auto ptr_field_ptr = ir_builder->CreateInBoundsGEP(span_type, span_val, std::vector<::llvm::Value*>{zero, offset});
  ir_builder->CreateStore(stack_val, ptr_field_ptr);

  DType param_dtype(DATA_EVAL_VALUE);
  auto span_v = NewValue(param_dtype.ToAbslSpan(), span_val, span_type);

  absl::StatusOr<ValuePtr> invoke_result;
  if (result_dtype.IsSimdVector()) {
    std::string fname = GetFunctionName(kBuiltinVectorEval, result_dtype.Elem());
    invoke_result = CallFunction(fname, {span_v});
  } else if (result_dtype.IsSimdColumnPtr()) {
    invoke_result = CallFunction(kBuiltinColumnEval, {span_v});
  } else {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("No func for dtype:{}", result_dtype)));
  }
  if (!invoke_result.ok()) {
    return invoke_result.status();
  }
  ValuePtr eval_result = invoke_result.value();
  if (assign_to) {
    auto assign_result = BuildIR(ctx, OP_ASSIGN, *assign_to, eval_result);
    if (!assign_result.ok()) {
      return assign_result.status();
    }
    eval_result = assign_result.value();
  }
  return eval_result;
}
}  // namespace llvm
}  // namespace rapidudf