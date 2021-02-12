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

def IOp : Yao_Op<"I", [NoSideEffect]> {
    let description = [{
        The `yao.I` creates an identity operator.
    }];

    let results = (outs AnyOperator:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

// ConstantLike
def HOp : Yao_Op<"H", [NoSideEffect]> {
    let description = [{
        The `yao.H` creates a hadamard operator.
    }];

    let results = (outs Operator<1>:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def XOp : Yao_Op<"X", [NoSideEffect]> {
    let description = [{
        The `yao.X` creates a Pauli X operator.
    }];

    let results = (outs Operator<1>:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def YOp : Yao_Op<"Y", [NoSideEffect]> {
    let description = [{
        The `yao.Y` creates a Pauli Y operator.
    }];

    let results = (outs Operator<1>:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def ZOp : Yao_Op<"Z", [NoSideEffect]> {
    let description = [{
        The `yao.Z` creates a Pauli Z operator.
    }];

    let results = (outs Operator<1>:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def CtrlFlagOp : Yao_Op<"create_flag", [NoSideEffect]> {
    let description = [{
        Create a qubit ctrl flag.
    }];

    // should be bool?
    let arguments = (ins Variadic<I64>:$inputs);
    let results = (outs AnyCtrlFlags:$res);
    
    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs) `:` type($res)
    }];
}

def LocationsOp : Yao_Op<"create_locations", [NoSideEffect]> {
    let description = [{
        Create a yao.locations.
    }];

    let arguments = (ins Variadic<Index>:$inputs);
    let results = (outs AnyLocations:$res);
    let assemblyFormat = [{
        $inputs attr-dict `:` type($inputs) `:` type($res)
    }];
}

def GateOp : Yao_Op<"gate", [NoSideEffect]> {
    let description = [{
        The `yao.gate` applys an operator at given gate locations.
    }];

    let arguments = (ins AnyOperator:$gate, AnyLocations:$locations);
    let assemblyFormat = [{
        $gate $locations attr-dict `:` type($gate) type($locations)
    }];
}

def CtrlOp : Yao_Op<"ctrl", [NoSideEffect]> {
    let description = [{
        The `yao.ctrl` applys an operator at given gate locations.
    }];

    let arguments = (ins AnyOperator:$gate, AnyLocations:$locations, AnyLocations:$ctrl_locations, AnyCtrlFlags:$flag);
    let assemblyFormat = [{
        $gate $locations $ctrl_locations $flag attr-dict `:` type($gate) type($locations) type($ctrl_locations) type($flag)
    }];
}

def MeasureOp : Yao_Op<"measure", [NoSideEffect]> {
    let description = [{
        The `yao.measure` measure at given locations.
    }];

    let arguments = (ins AnyLocations:$locations);
    let results = (outs AnyMeasureResult:$res);
    let assemblyFormat = [{
        $locations attr-dict `:` type($locations) `:` type($res)
    }];
}


#endif // YAO_OPS
