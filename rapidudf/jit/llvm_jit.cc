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
#include "rapidudf/jit/llvm_jit.h"
namespace rapidudf {
LLVMJitCompiler::LLVMJitCompiler() { Init(); }
void LLVMJitCompiler::Init() {
  // Open a new context and module.
  context_ = std::make_unique<llvm::LLVMContext>();
  module_ = std::make_unique<llvm::Module>("RapidUDF", *context_);
  // Create a new builder for the module.
  builder_ = std::make_unique<llvm::IRBuilder<>>(*context_);
}

// llvm::Function* LLVMJitCompiler::DefineFunction() {
//   llvm::FunctionPassManager ff;
//   std::vector<llvm::Type*> arg_types;
//   //   FunctionType *FT =
//   //       FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);
//   llvm::FunctionType* ftype = nullptr;
//   llvm::Function* f = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage, "Name", module_.get());

//   // Set names for all arguments.
//   for (auto& arg : f->args()) {
//     // arg.setName(const Twine &Name);
//   }
//   return nullptr;
//   //   unsigned Idx = 0;
//   //   for (auto &Arg : F->args())
//   //     Arg.setName(Args[Idx++]);

//   //   return F;
// }
void LLVMJitCompiler::Dump() {
  //   auto t = llvm::Type::getIntNTy(*context_, 4);
  //   printf("###%d\n", t->getIntegerBitWidth());

  //   auto sp = llvm::StructType::create(*context_, "a");
  //   sp->setBody(t);

  module_->print(llvm::errs(), nullptr);
}
}  // namespace rapidudf