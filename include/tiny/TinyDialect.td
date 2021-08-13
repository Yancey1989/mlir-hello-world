#ifndef TINY_DIALECT
#define TINY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Tiny_Dialect : Dialect {
    let name = "tiny";
    let summary = "a tiny dialect";
    let cppNamespace = "::mlir::tiny";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Tiny_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Tiny_Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT
