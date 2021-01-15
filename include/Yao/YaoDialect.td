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

//===----------------------------------------------------------------------===//
// Base yao operation definition.
//===----------------------------------------------------------------------===//

class Yao_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Yao_Dialect, mnemonic, traits>;

#endif // YAO_DIALECT
