//===- YaoDialect.cpp - Yao dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Yao/YaoDialect.h"
#include "Yao/YaoOps.h"

using namespace mlir;
using namespace mlir::yao;

//===----------------------------------------------------------------------===//
// Yao dialect.
//===----------------------------------------------------------------------===//

void YaoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Yao/YaoOps.cpp.inc"
      >();
}
