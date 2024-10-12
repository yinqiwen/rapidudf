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

#include <llvm/IR/Value.h>
#include <tuple>
#include <utility>
#include <vector>
#include "rapidudf/compiler/codegen.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/compiler/type.h"
#include "rapidudf/compiler/value.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace compiler {

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(const ast::RPN& rpn) {
  if (rpn.dtype.IsInvalid()) {
    RUDF_LOG_ERROR_STATUS(absl::InvalidArgumentError(fmt::format("Invalid rpn dtype to eval")));
  }
  std::vector<RPNEvalNode> eval_nodes;
  bool is_vector_expr = false;
  for (auto& node : rpn.nodes) {
    auto result = std::visit(
        [&](auto&& arg) -> absl::Status {
          using T = std::decay_t<decltype(arg)>;
          absl::StatusOr<ValuePtr> val_result;
          if constexpr (std::is_same_v<T, bool>) {
            val_result = BuildIR(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            val_result = BuildIR(arg);
          } else if constexpr (std::is_same_v<T, ast::VarDefine>) {
            val_result = BuildIR(arg);
          } else if constexpr (std::is_same_v<T, ast::VarAccessor>) {
            val_result = BuildIR(arg);
          } else if constexpr (std::is_same_v<T, ast::ConstantNumber>) {
            val_result = BuildIR(arg);
          } else if constexpr (std::is_same_v<T, ast::Array>) {
            val_result = BuildIR(arg);
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
            is_vector_expr = true;
          }
          eval_nodes.emplace_back(operand);
          return absl::OkStatus();
        },
        node);
    if (!result.ok()) {
      return result;
    }
  }

  if (is_vector_expr) {
    return BuildVectorIR(rpn.dtype, eval_nodes);
  }

  return BuildIR(rpn.dtype, eval_nodes);
}

absl::StatusOr<ValuePtr> JitCompiler::BuildIR(DType dtype, const std::vector<RPNEvalNode>& nodes) {
  std::vector<ValuePtr> operands;
  for (auto& node : nodes) {
    if (node.op != OP_INVALID) {
      auto op = node.op;
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
        result = codegen_->UnaryOp(op, operands[operands.size() - 1]);
        operands.pop_back();
      } else if (operand_count == 2) {
        result = codegen_->BinaryOp(op, operands[operands.size() - 2], operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
      } else if (operand_count == 3) {
        result = codegen_->TernaryOp(op, operands[operands.size() - 3], operands[operands.size() - 2],
                                     operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
        operands.pop_back();
      } else {
        RUDF_LOG_RETURN_FMT_ERROR("Unsupported op:{}", op);
      }
      if (!result.ok()) {
        return result.status();
      }
      operands.emplace_back(result.value());
    } else {
      operands.emplace_back(node.val);
    }
  }
  if (operands.size() != 1) {
    RUDF_LOG_RETURN_FMT_ERROR("After eval expr, {} operands rest.", operands.size());
  }
  return operands[0];
}

absl::Status JitCompiler::BuildVectorEvalIR(DType dtype, std::vector<RPNEvalNode>& nodes, ValuePtr cursor,
                                            ValuePtr remaining, ::llvm::Value* output) {
  using Operand = std::pair<DType, ::llvm::Value*>;
  std::vector<Operand> operands;
  auto normalize_operand_dtype = [&](int count) -> absl::StatusOr<DType> {
    DType compute_dtype;
    for (int i = 0; i < count; i++) {
      auto [dtype, _] = operands[operands.size() - count + i];
      if (dtype.IsSimdVector()) {
        compute_dtype = dtype.Elem();
        break;
      }
    }
    if (compute_dtype.IsInvalid()) {
      compute_dtype = operands[operands.size() - count].first;
    }
    for (int i = 0; i < count; i++) {
      auto [dtype, val] = operands[operands.size() - count + i];
      if (dtype.Elem() == compute_dtype) {
        continue;
      }
      auto result = codegen_->CastTo(val, dtype, compute_dtype);
      if (!result.ok()) {
        return result.status();
      }
      operands[operands.size() - count + i].first = compute_dtype.ToSimdVector();
      operands[operands.size() - count + i].second = result.value();
    }
    return compute_dtype;
  };
  for (size_t i = 0; i < nodes.size(); i++) {
    RPNEvalNode& node = nodes[i];
    OpToken op = node.op;
    if (op != OP_INVALID) {
      int operand_count = get_operand_count(op);
      DType dtype;
      absl::StatusOr<::llvm::Value*> result;
      if (operand_count == 1) {
        result = codegen_->UnaryOp(op, dtype, operands[operands.size() - 1].second);
        dtype = operands[operands.size() - 1].first;
        operands.pop_back();
      } else if (operand_count == 2) {
        auto normalize_result = normalize_operand_dtype(2);
        if (!normalize_result.ok()) {
          return normalize_result.status();
        }
        dtype = normalize_result.value();
        result =
            codegen_->BinaryOp(op, dtype, operands[operands.size() - 2].second, operands[operands.size() - 1].second);
        operands.pop_back();
        operands.pop_back();
      } else if (operand_count == 3) {
        auto normalize_result = normalize_operand_dtype(op == OP_CONDITIONAL ? 2 : 3);
        if (!normalize_result.ok()) {
          return normalize_result.status();
        }
        dtype = normalize_result.value();
        result = codegen_->TernaryOp(op, dtype, operands[operands.size() - 3].second,
                                     operands[operands.size() - 2].second, operands[operands.size() - 1].second);
        operands.pop_back();
        operands.pop_back();
        operands.pop_back();
      }
      if (result.ok()) {
        operands.emplace_back(std::make_pair(dtype, result.value()));
      } else {
        return result.status();
      }
    } else {
      auto value = node.val;
      if (value->GetDType().IsSimdVector()) {
        absl::StatusOr<std::pair<::llvm::Value*, ::llvm::Value*>> load_result;
        auto ptr_val = value->GetStructPtrValue().value();
        if (remaining) {
          load_result =
              codegen_->LoadNVector(value->GetDType().Elem(), ptr_val, cursor->LoadValue(), remaining->LoadValue());
        } else {
          load_result = codegen_->LoadVector(value->GetDType().Elem(), ptr_val, cursor->LoadValue());
        }
        if (!load_result.ok()) {
          return load_result.status();
        }
      } else {
        auto result = codegen_->NewConstVectorValue(value);
      }
      operands.emplace_back(std::make_pair(value->GetDType(), result.value().second));
    }
  }
  ::llvm::Value* eval_result = operands[0].second;

  if (remaining) {
    return codegen_->StoreNVector(dtype.Elem(), eval_result, output, cursor->LoadValue(), remaining->LoadValue());
  } else {
    return codegen_->StoreVector(dtype.Elem(), eval_result, output, cursor->LoadValue());
  }
}

absl::StatusOr<ValuePtr> JitCompiler::BuildVectorIR(DType result_dtype, std::vector<RPNEvalNode>& nodes) {
  OpToken last_op = nodes[nodes.size() - 1].op;
  std::optional<ValuePtr> assign_to;
  switch (last_op) {
    case OP_ASSIGN: {
      assign_to = nodes.front().val;
      // nodes.pop_front();
      nodes.erase(nodes.begin());
      nodes.pop_back();
      break;
    }
    case OP_PLUS_ASSIGN: {
      nodes[nodes.size() - 1].op = OP_PLUS;
      assign_to = nodes.front().val;
      break;
    }
    case OP_MOD_ASSIGN: {
      nodes[nodes.size() - 1].op = OP_MOD;
      assign_to = nodes.front().val;
      break;
    }
    case OP_MINUS_ASSIGN: {
      nodes[nodes.size() - 1].op = OP_MINUS;
      assign_to = nodes.front().val;
      break;
    }
    case OP_MULTIPLY_ASSIGN: {
      nodes[nodes.size() - 1].op = OP_MULTIPLY;
      assign_to = nodes.front().val;
      break;
    }
    case OP_DIVIDE_ASSIGN: {
      nodes[nodes.size() - 1].op = OP_DIVIDE;
      assign_to = nodes.front().val;
      break;
    }
    default: {
      break;
    }
  }

  // using ValidateOperandRef = std::pair<DType, int>;

  ValuePtr vector_size_val;
  // std::vector<ValidateOperandRef> operand_refs;
  for (size_t i = 0; i < nodes.size(); i++) {
    auto& node = nodes[i];
    OpToken op = node.op;
    if (op != OP_INVALID) {
    } else {
      auto value = node.val;
      // operand_refs.emplace_back(std::make_pair(value->GetDType(), static_cast<int>(i)));
      if (value->GetDType().IsSimdVector()) {
        auto result = value->GetVectorSizeValue();
        if (!result.ok()) {
          return result.status();
        }
        if (!vector_size_val) {
          vector_size_val = result.value();
        } else {
          // compare size
          auto cmp_result = codegen_->BinaryOp(OP_NOT_EQUAL, vector_size_val, result.value());
          if (!cmp_result.ok()) {
            return cmp_result.status();
          }
          auto condition = codegen_->NewCondition(0, false);
          codegen_->BeginIf(condition, cmp_result.value());
          auto status = ThrowVectorExprError("input vectors have different size");
          if (!status.ok()) {
            return status;
          }
          codegen_->EndIf(condition);
          codegen_->FinishCondition(condition);
        }
      }
    }
  }
  std::string new_vector_func_name = GetFunctionName(functions::kBuiltinNewSimdVector, result_dtype.Elem());
  auto result = codegen_->CallFunction(new_vector_func_name, {vector_size_val});
  if (!result.ok()) {
    return result.status();
  }

  auto output_val = codegen_->NewVar(result_dtype.ToSimdVector());
  auto status = output_val->CopyFrom(result.value());
  if (!status.ok()) {
    return status;
  }
  ::llvm::Value* output_ptr = output_val->GetStructPtrValue().value();

  auto vector_loop_limit_size = codegen_->BinaryOp(OP_MINUS, vector_size_val, codegen_->NewI32(k_vector_size)).value();
  auto cursor = codegen_->NewI32Var();
  auto loop = codegen_->NewLoop();
  auto cond = codegen_->BinaryOp(OP_LESS_EQUAL, cursor, vector_loop_limit_size).value();
  codegen_->AddLoopCond(loop, cond);
  status = BuildVectorEvalIR(result_dtype, nodes, cursor, nullptr, output_ptr);
  if (!status.ok()) {
    return status;
  }
  status = cursor->Inc(k_vector_size);
  if (!status.ok()) {
    return status;
  }
  codegen_->FinishLoop(loop);

  auto remaining_condition = codegen_->NewCondition(0, false);
  auto remaining_cond = codegen_->BinaryOp(OP_LESS, cursor, vector_size_val).value();
  codegen_->BeginIf(remaining_condition, remaining_cond);
  auto remain_n = codegen_->BinaryOp(OP_MINUS, vector_size_val, cursor).value();
  status = BuildVectorEvalIR(result_dtype, nodes, cursor, remain_n, output_ptr);
  if (!status.ok()) {
    return status;
  }
  codegen_->EndIf(remaining_condition);
  codegen_->FinishCondition(remaining_condition);

  return output_val;
}
}  // namespace compiler
}  // namespace rapidudf