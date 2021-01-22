// RUN: yao-opt %s | yao-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        // CHECK: %{{.*}} = yao.H : !yao.gate<1>
        %H = yao.H : !yao.gate<1>
        %I = yao.I : !yao.gate<1>
        //%H2 = yao.chain(%H, %H) : !yao.gate<1>, !yao.gate<1> : !yao.gate<1>
        return
    }
}
