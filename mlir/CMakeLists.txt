add_mlir_dialect_library(MLIRTiny
        TinyDialect.cpp
        TinyOps.cpp
        MLIRGen.cpp
        LowerToAffine.cpp
        LowerToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/tiny

        DEPENDS
        MLIRTinyOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)