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
def AnyGate : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::GateType>()">, "gate">;

// Any integer type (regardless of signedness semantics) of a specific width.
class Gate<int width>
    : DialectType<Yao_Dialect, CPred<"$_self.isa<::mlir::yao::GateType>() && $_self.cast<::mlir::yao::GateType>().getNumQubits() == " # width>, width # "-bit gate"> {
  int bitwidth = width;
}

//===----------------------------------------------------------------------===//
// Base yao operation definition.
//===----------------------------------------------------------------------===//

class Yao_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Yao_Dialect, mnemonic, traits>;

#endif // YAO_DIALECT
