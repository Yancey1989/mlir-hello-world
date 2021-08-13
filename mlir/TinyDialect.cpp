#include "tiny/TinyDialect.h"
#include "tiny/TinyOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tiny;

#include "tiny/TinyOpsDialect.cpp.inc"

void TinyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny/TinyOps.cpp.inc"
      >();
}
