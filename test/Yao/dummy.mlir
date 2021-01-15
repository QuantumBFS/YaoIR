// RUN: yao-opt %s | yao-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = yao.foo %{{.*}} : i32
        %res = yao.foo %0 : i32
        return
    }
}
