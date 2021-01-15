add_mlir_dialect_library(MLIRYao
        YaoDialect.cpp
        YaoOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Yao

        DEPENDS
        MLIRYaoOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
