#include "tiny/Parser.h"
#include "tiny/TinyDialect.h"
#include "tiny/MLIRGen.h"
#include "tiny/Passes.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace tiny;

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tiny file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
enum Action { None, DumpAST, DumpMLIR, DumpMLIRLLVM, RunJIT};

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
               cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the LLVM dump")),
               cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoke the main function")));


std::unique_ptr<tiny::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int dumpMLIR(tiny::ModuleAST& module) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tiny::TinyDialect>();
    mlir::OwningModuleRef m = mlirGen(context, module);
    m->dump();
    return 0;
}

int dumpMLIRLLVM(tiny::ModuleAST& module) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tiny::TinyDialect>();
    mlir::OwningModuleRef m = mlirGen(context, module);

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    {
        mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

        // Partially lower the tiny dialect with a few cleanups afterwards.
        optPM.addPass(mlir::tiny::createLowerToAffinePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }


    // add LowerToLLVMPass
    //pm.addPass(mlir::tiny::createLowerToLLVMPass());
    if (mlir::failed(pm.run(*m)))
        return -1;
    return 0;
}

int main(int argc, char** argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "tiny compiler!\n");
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;
    switch(emitAction) {
    case Action::DumpAST:
        tiny::dump(*moduleAST);
        return 0;
    case Action::DumpMLIR:
        return dumpMLIR(*moduleAST);
    case Action::DumpMLIRLLVM:
        return dumpMLIRLLVM(*moduleAST);
    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }
    return 0;
}