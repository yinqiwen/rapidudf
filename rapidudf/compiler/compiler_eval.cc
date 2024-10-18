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

#include <tuple>
#include <utility>
#include <vector>
#include "llvm/IR/Value.h"
#include "rapidudf/compiler/codegen.h"
#include "rapidudf/compiler/compiler.h"
#include "rapidudf/compiler/type.h"
#include "rapidudf/compiler/value.h"
#include "rapidudf/functions/functions.h"
#include "rapidudf/functions/names.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/simd/vector.h"

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
  using Operand = std::pair<::llvm::Value*, ::llvm::Value*>;
  std::vector<Operand> operands;
  for (size_t i = 0; i < nodes.size(); i++) {
    RPNEvalNode& node = nodes[i];
    OpToken op = node.op;
    if (op != OP_INVALID) {
      int operand_count = get_operand_count(op);
      DType compute_dtype = node.op_compute_dtype;
      absl::StatusOr<::llvm::Value*> result;
      bool use_vector_call = functions::has_vector_buitin_func(op, compute_dtype);
      if (use_vector_call && !codegen_->IsExternFunctionExist(GetFunctionName(op, compute_dtype.ToSimdVector()))) {
        use_vector_call = false;
      }

      if (operand_count == 1) {
        // result = codegen_->UnaryOp(op, compute_dtype, operands[operands.size() - 1].second);
        if (use_vector_call) {
          result = codegen_->VectorUnaryOp(op, compute_dtype, operands[operands.size() - 1].first, node.op_temp_val);
        } else {
          result = codegen_->UnaryOp(op, compute_dtype, operands[operands.size() - 1].second);
        }
      } else if (operand_count == 2) {
        if (use_vector_call) {
          result = codegen_->VectorBinaryOp(op, compute_dtype, operands[operands.size() - 2].first,
                                            operands[operands.size() - 1].first, node.op_temp_val);
        } else {
          result = codegen_->BinaryOp(op, compute_dtype, operands[operands.size() - 2].second,
                                      operands[operands.size() - 1].second);
        }
      } else if (operand_count == 3) {
        if (use_vector_call) {
          result = codegen_->VectorTernaryOp(op, compute_dtype, operands[operands.size() - 3].first,
                                             operands[operands.size() - 2].first, operands[operands.size() - 1].first,
                                             node.op_temp_val);
        } else {
          result = codegen_->TernaryOp(op, compute_dtype, operands[operands.size() - 3].second,
                                       operands[operands.size() - 2].second, operands[operands.size() - 1].second);
        }
      }
      for (int j = 0; j < operand_count; j++) {
        operands.pop_back();
      }
      if (result.ok()) {
        if (!use_vector_call) {
          codegen_->Store(result.value(), node.op_temp_val);
        }
        operands.emplace_back(std::make_pair(node.op_temp_val, result.value()));
      } else {
        return result.status();
      }
    } else {
      auto value = node.val;
      ::llvm::Value* load_value_ptr = nullptr;
      ::llvm::Value* load_value = nullptr;
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
        load_value_ptr = load_result.value().first;
        load_value = load_result.value().second;
      } else {
        load_value_ptr = node.constant_vector_val_ptr;
        load_value = node.constant_vector_val;
      }
      operands.emplace_back(std::make_pair(load_value_ptr, load_value));
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
  ValuePtr assign_to;
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

  ValuePtr output_val;
  if (nodes.size() > 1) {
    using Operand = std::pair<DType, std::vector<size_t>>;
    std::vector<Operand> operands;
    ValuePtr vector_size_val;
    auto normalize_operand_dtype = [&](OpToken op, std::vector<size_t>& idxs) -> absl::StatusOr<DType> {
      int operand_count = get_operand_count(op);
      int count = operand_count;
      if (op == OP_CONDITIONAL) {
        count = 2;
      }
      DType compute_dtype;
      bool no_vector = true;
      for (int i = 0; i < count; i++) {
        auto [dtype, _] = operands[operands.size() - count + i];
        if (dtype.IsSimdVector()) {
          compute_dtype = dtype.Elem();
          no_vector = false;
          break;
        }
      }
      if (compute_dtype.IsInvalid()) {
        compute_dtype = operands[operands.size() - count].first;
      }

      for (int i = 0; i < count; i++) {
        auto [dtype, prev_idxs] = operands[operands.size() - count + i];
        idxs.insert(idxs.end(), prev_idxs.begin(), prev_idxs.end());

        if (dtype.Elem() == compute_dtype) {
          continue;
        }
        for (auto idx : prev_idxs) {
          auto val = nodes[idx].val;

          auto result = codegen_->CastTo(val, compute_dtype);
          if (!result.ok()) {
            return result.status();
          }
          nodes[idx].val = result.value();
        }
        operands[operands.size() - count + i].first = compute_dtype.ToSimdVector();
      }
      for (int i = 0; i < operand_count; i++) {
        operands.pop_back();
      }
      if (no_vector) {
        return compute_dtype.Elem();
      } else {
        return compute_dtype.ToSimdVector();
      }
    };

    for (size_t i = 0; i < nodes.size(); i++) {
      auto& node = nodes[i];
      OpToken op = node.op;
      if (op != OP_INVALID) {
        DType dtype;
        std::vector<size_t> idxs;
        auto normalize_result = normalize_operand_dtype(op, idxs);
        if (!normalize_result.ok()) {
          return normalize_result.status();
        }
        dtype = normalize_result.value();
        if (is_compare_op(op)) {
          operands.emplace_back(std::make_pair(DATA_BIT, idxs));
        } else {
          operands.emplace_back(std::make_pair(dtype, idxs));
        }
        node.op_compute_dtype = dtype.Elem();
        if (is_compare_op(op)) {
          node.op_temp_val = codegen_->NewVectorVar(DATA_BIT);

        } else {
          node.op_temp_val = codegen_->NewVectorVar(dtype);
        }

      } else {
        auto value = node.val;
        operands.emplace_back(std::make_pair(value->GetDType(), std::vector<size_t>{i}));
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
    for (size_t i = 0; i < nodes.size(); i++) {
      auto& node = nodes[i];
      if (node.op != OP_INVALID) {
        continue;
      }
      auto value = node.val;
      if (!value->GetDType().IsSimdVector()) {
        auto result = codegen_->NewStackConstantVector(value);
        if (!result.ok()) {
          return result.status();
        }
        node.constant_vector_val_ptr = result.value().first;
        node.constant_vector_val = result.value().second;
      }
    }

    std::string new_vector_func_name = GetFunctionName(functions::kBuiltinNewSimdVector, result_dtype.Elem());
    auto result = codegen_->CallFunction(new_vector_func_name, {vector_size_val});
    if (!result.ok()) {
      return result.status();
    }

    output_val = codegen_->NewVar(result_dtype.ToSimdVector());
    auto status = output_val->CopyFrom(result.value());
    if (!status.ok()) {
      return status;
    }
    ::llvm::Value* output_ptr = output_val->GetStructPtrValue().value();

    auto vector_loop_limit_size =
        codegen_->BinaryOp(OP_MINUS, vector_size_val, codegen_->NewI32(simd::kVectorUnitSize)).value();
    auto cursor = codegen_->NewI32Var();
    auto loop = codegen_->NewLoop();
    auto cond = codegen_->BinaryOp(OP_LESS_EQUAL, cursor, vector_loop_limit_size).value();
    codegen_->AddLoopCond(loop, cond);
    status = BuildVectorEvalIR(result_dtype, nodes, cursor, nullptr, output_ptr);
    if (!status.ok()) {
      return status;
    }
    status = cursor->Inc(simd::kVectorUnitSize);
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
  } else {
    output_val = nodes[0].val;
  }

  if (assign_to) {
    auto status = assign_to->CopyFrom(output_val);
    if (!status.ok()) {
      return status;
    }
    return assign_to;
  }
  return output_val;
}
}  // namespace compiler
}  // namespace rapidudf