//===- YaoDialect.td - Yao dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef YAO_DIALECT
#define YAO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Yao dialect definition.
//===----------------------------------------------------------------------===//

def Yao_Dialect : Dialect {
    let name = "yao";
    let summary = "Yao a Quantum MLIR dialect.";
    let description = [{
    }];
    let cppNamespace = "::mlir::yao";
}


// Any integer type irrespective of its width and signedness semantics.
def AnyOperator : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::OperatorType>()">, "operator">;

// Any integer type (regardless of signedness semantics) of a specific width.
class Operator<int width>
    : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::OperatorType>() && $_self.cast<::mlir::yao::OperatorType>().getNumQubits() == " # width>, width # "-bit operator"> {
  int bitwidth = width;
}

def AnyCtrlFlags : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::CtrlFlagType>()">, "ctrlflags">;

class CtrlFlags<int width>
    : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::CtrlFlagType() && $_self.cast<::mlir::yao::CtrlFlagType().getNumQubits() == " # width>, width # "-bit ctrlflags"> {
  int nqubits = width;
}

def AnyLocations : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::LocationsType>()">, "locations">;

class Locations<int width>
    : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::LocationsType() && $_self.cast<::mlir::yao::LocationsType().getNumQubits() == " # width>, width # "-bit locations"> {
  int nqubits = width;
}

def AnyMeasureResult : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::MeasureResultType>()">, "measure_result">;

class MeasureResult<int width>
    : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::MeasureResultType() && $_self.cast<::mlir::yao::MeasureResultType().getNumQubits() == " # width>, width # "-bit measure_result"> {
  int nqubits = width;
}

//===----------------------------------------------------------------------===//
// Base yao operation definition.
//===----------------------------------------------------------------------===//

class Yao_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Yao_Dialect, mnemonic, traits>;

#endif // YAO_DIALECT
