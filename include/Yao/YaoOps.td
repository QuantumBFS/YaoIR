//===- YaoOps.td - Yao dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef YAO_OPS
#define YAO_OPS

include "YaoDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IOp : Yao_Op<"I", [ConstantLike, NoSideEffect]> {
    let description = [{
        The `yao.I` operation performs a hadamard of one qubit
    }];

    let results = (outs AnyGate:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def HOp : Yao_Op<"H", [ConstantLike, NoSideEffect]> {
    let description = [{
        The `yao.H` operation performs a hadamard of one qubit
    }];

    let results = (outs Gate<1>:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def ChainOp : Yao_Op<"chain", [NoSideEffect]> {
    let description = [{
        The `yao.chain` operation performs a combination of successive gate operations
    }];

    let arguments = (ins Variadic<AnyGate>:$inputs);
    let results = (outs AnyGate:$res);

    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs) `:` type($res)
    }];
}
#endif // YAO_OPS
