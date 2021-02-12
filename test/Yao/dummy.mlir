// RUN: yao-opt %s | yao-opt | FileCheck %s

module {
    // CHECK: func @bar()
    func @bar() {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %c3 = constant 3 : index
        %c4 = constant 4 : index
        %c5 = constant 5 : index

        %l11 = yao.create_locations %c2 : index : !yao.locations<1>
        %l12 = yao.create_locations %c3 : index : !yao.locations<1>
        %l21 = yao.create_locations %c2, %c3 : index, index : !yao.locations<2>
        %l31 = yao.create_locations %c2, %c3, %c4 : index, index, index : !yao.locations<3>

        %H = yao.H : !yao.operator<1>
        %I = yao.I : !yao.operator<1>
        %flag1 = yao.create_flag %c0 : index, !yao.ctrlflag<1>
        %flag3 = yao.create_flag %c0, %c1, %c0 : index, index, index : !yao.ctrlflag<3>
        // 1. single qubit gate
        yao.gate %H, %l11 : !yao.operator<1> !yao.locations<1>
        // 2. single qubit ctrl gate
        yao.ctrl %H, %l11, %l12, %flag1 : !yao.operator<1> !yao.locations<1> !yao.locations<1> !yao.ctrlflag<1>
        // 3. measure
        %C1 = yao.measure %l21 : !yao.locations<2> : !yao.measure_result<2>
        // 4. multi-qubit gate
        yao.gate %M, %l31 : !yao.operator<3> !yao.locations<3>
        // 5. multi-ctrl single qubit gate
        yao.ctrl %H, %l11, %l31 : !yao.operator<1> !yao.locations<1> !yao.locations<3>
        // 6. multi-ctrl multi-qubit gate
        yao.ctrl %S, %l21, %l31, %flag3 : !yao.operator<2> !yao.locations<2> !yao.locations<3> !yao.ctrlflag<3>
        return
    }
}
