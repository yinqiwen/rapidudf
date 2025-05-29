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
#include <memory>
#include "absl/status/statusor.h"
#include "rapidudf/meta/dtype.h"

namespace llvm {
class IRBuilderBase;
class Value;
class Type;
class Constant;
}  // namespace llvm

namespace rapidudf {
namespace compiler {

class Value;
using ValuePtr = std::shared_ptr<Value>;
class Value : public std::enable_shared_from_this<Value> {
 private:
  struct Private {
    explicit Private() = default;
  };

 public:
  Value(Private, DType dtype, ::llvm::IRBuilderBase* ir_builder, ::llvm::Value* val, ::llvm::Type* ptr_element_type);

  static ValuePtr New(DType dtype, ::llvm::IRBuilderBase* ir_builder, ::llvm::Value* val,
                      ::llvm::Type* ptr_element_type = nullptr) {
    return std::make_shared<Value>(Private(), dtype, ir_builder, val, ptr_element_type);
  }
  ~Value() {}

  ::llvm::Value* LoadValue();
  ::llvm::Value* GetPtrValue();

  absl::StatusOr<::llvm::Value*> GetStructPtrValue();
  absl::StatusOr<::llvm::Value*> GetStructSizeValue();
  absl::StatusOr<ValuePtr> GetRawVectorSizeValue();
  absl::StatusOr<ValuePtr> GetVectorArrowFlag();

  DType GetDType() { return dtype_; }
  bool IsWritable() const;

  absl::Status CopyFrom(ValuePtr other);

  absl::Status Inc(uint64_t v);

 private:
  ValuePtr SelfPtr() { return shared_from_this(); }

  DType dtype_;
  ::llvm::IRBuilderBase* ir_builder_ = nullptr;
  ::llvm::Value* val_ = nullptr;
  ::llvm::Type* ptr_element_type_ = nullptr;

  std::vector<uint64_t> const_values_;
};
}  // namespace compiler
}  // namespace rapidudf