/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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

#pragma once
#include <memory>
#include "absl/status/status.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/optype.h"
namespace rapidudf {
namespace llvm {

class JitCompiler;
class Value;
using ValuePtr = std::shared_ptr<Value>;
class Value : public std::enable_shared_from_this<Value> {
 private:
  struct Private {
    explicit Private() = default;
  };

 public:
  Value(Private, DType dtype, JitCompiler* c, ::llvm::Value* val, const std::string& name);

  static ValuePtr New(DType dtype, JitCompiler* c, ::llvm::Value* val, const std::string& name = "") {
    return std::make_shared<Value>(Private(), dtype, c, val, name);
  }
  ~Value() {}

  const std::string& GetName() const { return name_; }
  ::llvm::Value* GetValue() { return val_; }
  DType GetDType() { return dtype_; }

  ValuePtr UnaryOp(OpToken op);
  ValuePtr BinaryOp(OpToken op, ValuePtr right);
  ValuePtr Select(ValuePtr true_val, ValuePtr false_val);

  ValuePtr CastTo(DType dtype);
  absl::Status CopyFrom(ValuePtr other);

 private:
  ValuePtr SelfPtr() { return shared_from_this(); }

  DType dtype_;
  JitCompiler* compiler_ = nullptr;
  ::llvm::IRBuilder<>* ir_builder_ = nullptr;
  ::llvm::Value* val_ = nullptr;

  std::string name_;

  std::vector<uint64_t> const_values_;
};
}  // namespace llvm
}  // namespace rapidudf