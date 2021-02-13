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
  addTypes<OperatorType, CtrlFlagType, LocationsType, MeasureResultType>();
}

/// Parse an instance of a type registered to the dialect.
Type YaoDialect::parseType(DialectAsmParser &parser) const {
  std::string found = "";
  if (!parser.parseOptionalKeyword("operator")) {
    found = "operator";
  // all the op stuff here    
  } else if (!parser.parseOptionalKeyword("locations")) {
    found = "locations";
  } else if (!parser.parseOptionalKeyword("ctrlflags")) {
    found = "ctrlflags";
  } else if (!parser.parseOptionalKeyword("measure_result")) {
    found = "measure_result";
  } else {
    // parser.parseKeyword("measure_result");
    return Type();
  }

  if (parser.parseLess())
    return Type();
  size_t number = 0;
  if (parser.parseInteger(number)) return Type();
  if (number == 0) {
    parser.emitError(parser.getCurrentLocation(), "operator must have at least size one ");
    return Type();
  }

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  
  llvm::errs() << "found " << found;
  if (found == "operator")
    return OperatorType::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), number);
  if (found == "locations")
    return LocationsType::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), number);
  if (found == "ctrlflags")
    return CtrlFlagType::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), number);
  if (found == "measure_result")
    return MeasureResultType::getChecked(parser.getEncodedSourceLoc(parser.getCurrentLocation()), number);
  llvm_unreachable("unknown yao type");
}

/// Print an instance of a type registered to the dialect.
void YaoDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto operatorType = type.dyn_cast<OperatorType>())
    printer << "operator<" << operatorType.getNumQubits() << ">";
  else if (auto operatorType = type.dyn_cast<LocationsType>())
    printer << "locations<" << operatorType.getNumQubits() << ">";
  else if (auto operatorType = type.dyn_cast<CtrlFlagType>())
    printer << "ctrlflags<" << operatorType.getNumQubits() << ">";
  else if (auto operatorType = type.dyn_cast<MeasureResultType>())
    printer << "measure_result<" << operatorType.getNumQubits() << ">";
  else llvm_unreachable("unknown yao type");
}