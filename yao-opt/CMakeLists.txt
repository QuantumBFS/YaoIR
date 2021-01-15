get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRYao
        )
add_llvm_executable(yao-opt yao-opt.cpp)

llvm_update_compile_flags(yao-opt)
target_link_libraries(yao-opt PRIVATE ${LIBS})
