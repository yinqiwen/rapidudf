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
#include <optional>
#include <variant>
#include <vector>

#include "rapidudf/ast/expression.h"
#include "rapidudf/builtin/builtin_symbols.h"
#include "rapidudf/jit/llvm/codegen.h"
#include "rapidudf/jit/llvm/jit.h"
#include "rapidudf/jit/llvm/jit_session.h"
#include "rapidudf/jit/llvm/type.h"
#include "rapidudf/jit/llvm/value.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/operand.h"
#include "rapidudf/meta/optype.h"

namespace rapidudf {
namespace llvm {
struct IREvalValue {
  DType dtype;
  ::llvm::Value* element_dtype = nullptr;
  ::llvm::Value* vector_size = nullptr;
  ::llvm::Value* value = nullptr;  // vector ptr or scalar
  ::llvm::Value* ptr = nullptr;    // vector ptr or scalar
  OpToken op = OP_INVALID;
};

absl::StatusOr<ValuePtr> JitCompiler::BuildVectorEvalIR2(FunctionCompileContextPtr ctx, DType result_dtype,
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
  CodeGen codegen(ir_builder, GetSession()->label_cursor);

  bool has_column = false;
  ::llvm::Value* output = nullptr;

  ::llvm::Value* vector_size_val = nullptr;
  std::vector<IREvalValue> ir_eval_vals;
  ValuePtr ret_value;
  for (auto& node : nodes) {
    IREvalValue eval_val;
    const OpToken* op_ptr = std::get_if<OpToken>(&node);
    if (op_ptr != nullptr) {
      eval_val.op = *op_ptr;
    } else {
      auto value = std::get<ValuePtr>(node);
      eval_val.dtype = value->GetDType();
      if (value->GetDType().IsSimdVector()) {
        ret_value = value;
        DType ele_dtype = value->GetDType().Elem();
        eval_val.element_dtype = ir_builder->getInt64(ele_dtype.Control());
        auto size_result = value->GetVectorSizeValue();
        if (!size_result.ok()) {
          return size_result.status();
        }
        eval_val.vector_size = size_result.value();
        auto ptr_value = value->GetStructPtrValue();
        eval_val.value = ptr_value.value();
        output = ptr_value.value();
        if (vector_size_val == nullptr) {
          vector_size_val = eval_val.vector_size;
        } else {
          // todo cmp size
        }
      } else if (value->GetDType().IsSimdColumnPtr()) {
        has_column = true;
        // todo
      } else {
        // stringView
        auto val_result = codegen.NewConstVectorValue(eval_val.dtype, value->GetValue());
        if (!val_result.ok()) {
          return val_result.status();
        }
        eval_val.value = val_result.value();
      }
    }
    ir_eval_vals.emplace_back(eval_val);
  }
  if (vector_size_val == nullptr) {
    return absl::InvalidArgumentError("No vector size to do later vector compute");
  }
  ::llvm::Value* vector_loop_limit = ir_builder->CreateSub(vector_size_val, ir_builder->getInt32(k_vector_size));
  std::string print_size = "print_size";
  // CallFunction(print_size, {NewValue(DATA_I32, vector_loop_limit)});

  ::llvm::Value* cursor_val = ir_builder->CreateAlloca(ir_builder->getInt32Ty());
  ir_builder->CreateStore(ir_builder->getInt32(0), cursor_val);

  uint32_t label_cursor = GetLabelCursor();
  std::string loop_cond_label = fmt::format("loop_cond_{}", label_cursor);
  std::string loop_body_label = fmt::format("loop_body_{}", label_cursor);
  std::string loop_end_label = fmt::format("loop_end_{}", label_cursor);
  auto* current_func = ir_builder->GetInsertBlock()->getParent();
  ::llvm::BasicBlock* loop_cond_block =
      ::llvm::BasicBlock::Create(ir_builder->getContext(), loop_cond_label, current_func);
  ::llvm::BasicBlock* loop_end_block = ::llvm::BasicBlock::Create(ir_builder->getContext(), loop_end_label);
  ::llvm::BasicBlock* loop_body_block = ::llvm::BasicBlock::Create(ir_builder->getContext(), loop_body_label);

  ir_builder->CreateBr(loop_cond_block);
  ir_builder->SetInsertPoint(loop_cond_block);
  auto loaded_cursor = ir_builder->CreateLoad(ir_builder->getInt32Ty(), cursor_val);
  auto cond_val = ir_builder->CreateICmpSLE(loaded_cursor, vector_loop_limit);
  ir_builder->CreateCondBr(cond_val, loop_body_block, loop_end_block);

  loop_body_block->insertInto(current_func);
  ir_builder->SetInsertPoint(loop_body_block);
  std::string xx = "test_extern_func";
  // CallFunction(xx, {});

  std::vector<::llvm::Value*> operands;
  for (size_t i = 0; i < ir_eval_vals.size(); i++) {
    IREvalValue& node = ir_eval_vals[i];
    if (node.op == OP_INVALID) {
      if (node.dtype.IsSimdVector()) {
        auto load_result = codegen.LoadVector(node.dtype, node.value, loaded_cursor);
        if (!load_result.ok()) {
          return load_result.status();
        }
        operands.emplace_back(load_result.value());
      } else if (node.dtype.IsSimdColumnPtr()) {
        // todo
      } else {
        operands.emplace_back(node.value);
        // codegen.NewConstVectorValue(DType dtype, ::llvm::Constant *val)
      }
    } else {
      int operand_count = get_operand_count(node.op);
      DType dtype(DATA_I32);
      absl::StatusOr<::llvm::Value*> result;
      if (operand_count == 1) {
        result = codegen.UnaryOp(node.op, dtype, operands[operands.size() - 1]);
        operands.pop_back();
      } else if (operand_count == 2) {
        result = codegen.BinaryOp(node.op, dtype, operands[operands.size() - 2], operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
      } else if (operand_count == 3) {
        result = codegen.TernaryOp(node.op, dtype, operands[operands.size() - 3], operands[operands.size() - 2],
                                   operands[operands.size() - 1]);
        operands.pop_back();
        operands.pop_back();
        operands.pop_back();
      }
      if (result.ok()) {
        operands.emplace_back(result.value());
      } else {
        return result.status();
      }
    }
  }
  ::llvm::Value* result = operands[0];
  DType dtype(DATA_I32);
  codegen.StoreVector(dtype, result, output, loaded_cursor);
  auto new_cursor_val = ir_builder->CreateAdd(loaded_cursor, ir_builder->getInt32(k_vector_size));
  ir_builder->CreateStore(new_cursor_val, cursor_val);
  ir_builder->CreateBr(loop_cond_block);
  loop_end_block->insertInto(current_func);
  ir_builder->SetInsertPoint(loop_end_block);

  irb

      uint32_t remain_label_cursor = GetLabelCursor();
  std::string if_block_label = fmt::format("if_block_{}", remain_label_cursor);
  std::string if_end_label = fmt::format("continue_{}", remain_label_cursor);
  loaded_cursor = ir_builder->CreateLoad(ir_builder->getInt32Ty(), cursor_val);
  auto remain = ir_builder->CreateICmpSLT(loaded_cursor, vector_size_val);
  ::llvm::BasicBlock* if_block = ::llvm::BasicBlock::Create(ir_builder->getContext(), if_block_label);
  ::llvm::BasicBlock* if_end_block = ::llvm::BasicBlock::Create(ir_builder->getContext(), if_end_label);
  ir_builder->CreateCondBr(remain, if_block, if_end_block);
  if_block->insertInto(current_func);
  ir_builder->SetInsertPoint(if_block);
  ::llvm::Value* remain_n = ir_builder->CreateSub(vector_size_val, loaded_cursor);
  std::vector<::llvm::Value*> remain_operands;
  for (size_t i = 0; i < ir_eval_vals.size(); i++) {
    IREvalValue& node = ir_eval_vals[i];
    if (node.op == OP_INVALID) {
      if (node.dtype.IsSimdVector()) {
        auto load_result = codegen.LoadNVector(node.dtype, node.value, loaded_cursor, remain_n);
        if (!load_result.ok()) {
          return load_result.status();
        }
        remain_operands.emplace_back(load_result.value());
      } else if (node.dtype.IsSimdColumnPtr()) {
        // todo
      } else {
        remain_operands.emplace_back(node.value);
        // codegen.NewConstVectorValue(DType dtype, ::llvm::Constant *val)
      }
    } else {
      int operand_count = get_operand_count(node.op);
      DType dtype(DATA_I32);
      absl::StatusOr<::llvm::Value*> result;
      if (operand_count == 1) {
        result = codegen.UnaryOp(node.op, dtype, remain_operands[remain_operands.size() - 1]);
        remain_operands.pop_back();
      } else if (operand_count == 2) {
        result = codegen.BinaryOp(node.op, dtype, remain_operands[remain_operands.size() - 2],
                                  remain_operands[remain_operands.size() - 1]);
        remain_operands.pop_back();
        remain_operands.pop_back();
      } else if (operand_count == 3) {
        result =
            codegen.TernaryOp(node.op, dtype, remain_operands[remain_operands.size() - 3],
                              remain_operands[remain_operands.size() - 2], remain_operands[remain_operands.size() - 1]);
        remain_operands.pop_back();
        remain_operands.pop_back();
        remain_operands.pop_back();
      }
      if (result.ok()) {
        remain_operands.emplace_back(result.value());
      } else {
        return result.status();
      }
    }
  }
  ::llvm::Value* remain_result = remain_operands[0];
  codegen.StoreNVector(dtype, remain_result, output, loaded_cursor, remain_n);
  // CallFunction(xx, {});
  ir_builder->CreateBr(if_end_block);

  if_end_block->insertInto(current_func);
  ir_builder->SetInsertPoint(if_end_block);

  return ret_value;
}
}  // namespace llvm
}  // namespace rapidudf