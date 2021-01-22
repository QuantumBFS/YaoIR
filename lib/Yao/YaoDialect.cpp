//===- YaoDialect.cpp - Yao dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Yao/YaoDialect.h"
#include "Yao/YaoOps.h"
#include "Yao/YaoTypes.h"
#include "mlir/IR/DialectImplementation.h"

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

  //llvm::errs() << "adding gate type\n";
  addTypes<GateType>();
}

/// Parse an instance of a type registered to the dialect.
Type YaoDialect::parseType(DialectAsmParser &parser) const {
  llvm::errs() << "starting parse\n";
  if (parser.parseKeyword("gate"))
    return Type();
  llvm::errs() << "parsed gate kw\n";
  
  if (parser.parseLess())
    return Type();
  llvm::errs() << "parsed less kw\n";
  size_t number = 0;
  if (parser.parseInteger(number)) return Type();
  llvm::errs() << "parsed num kw\n";
  if (number == 0) {
    parser.emitError(parser.getCurrentLocation(), "gate must apply to at least one qubit ");
    return Type();
  }

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  llvm::errs() << "parsed gt kw\n";
  return GateType::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), number);
}

/// Print an instance of a type registered to the dialect.
void YaoDialect::printType(Type type, DialectAsmPrinter &printer) const {
  GateType gateType = type.cast<GateType>();
  printer << "gate<" << gateType.getNumQubits() << ">";
}