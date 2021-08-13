# MLIR--Hello World

最近的一些工作会接触到 AI 编译器相关的技术，其中 MLIR 是 Google 在 2019 年的 GCP 上提出的一种新的 AI 编译基础设施,
其目的是希望现有的编译优化技术（例如 XLA、TVM等）能够以 Dialect 的方式接入进来，从而使各家的编译优化技术能够一起发挥效用。

和大部分程序语言的 101 一样，本文尝试实现一种新的语言 **Tiny**，实现打印字符串 `Hello World!` 的功能。 为了快速理解 MLIR 的
工作机制，本例中省却了大量优化的的 Pass 实现，而这些 Pass 才是一个编译器的灵魂所在，想要了解更详细的例子，请参考官网的
[Tutorials](https://mlir.llvm.org/docs/Tutorials/).

## Prerequirements

1. 编译 LLVM-MLIR 并安装

  ``` text
  cd mlir-hello-world/llvm-project
  mkdir build

  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_INSTALL_PREFIX=$PWD/../install
    
  ninja install -j8
  ```

1. 设置环境变量

  ``` text
  cd mlir-hello-world
  export PREFIX=$PWD/llvm-project/install
  export BUILD_DIR=$PWD/llvm-project/build 
  ```

1. 编译 Tiny

  ``` text
  cd mlir-hello-world
  mkdir build && cd build
  cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
  cmake --build . --target tinyc
  ```

## 语法解析：Tiny Language 到 Tiny AST

编写如下打印 `Hello World!` 的程序并保存为 `hello.tiny`：

``` text
def  main() {
  print("Hello world!")
}
```

这其中包括两个关键词:

1. `def` 用来定义一个 main 函数，作为程序执行的入口。
2. `print` 用来打印一个字符串。

本例在 Toy Parser 基础上稍作修改增加了 String Expr, 用来做 tiny 语言的语法解析工具，代码见
[Tiny Parser](./include/tiny/Parser.h)您可以使用如下命令查看生成的 AST：

``` text
$ echo 'def main() {print("Hello World!");}' | ./bin/tinyc -emit ast
  Module:
    Function
      Proto 'main' @hello.tiny:1:1
      Params: []
      Block {
        Print [ @hello.tiny:2:3
          Literal: <14>[ 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 92, 110] @hello.tiny:2:9
        ]
      } // Block
```

因为目前 MLIR 还没有一条完备的通路可以支持 String 类型，所以在 AST 这里用 Integer Tensor 来表示 String 类型。

## 实现 Tiny Dialect: Tiny AST 到 Tiny IR

MLIR 允许用户使用更简洁的 `TableGen` 语法来定义 Dialect 以及其对应的 Operations。 通过继承 Dialect 可以
很容易的新增一个 `TinyDialect`:

``` cpp
def Tiny_Dialect : Dialect {
  let name = "tiny";
  let cppNamespace = "::mlir::tiny";
}
```

添加 `print` Operation:

``` cpp
class Tiny_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// Print Operation
def PrintOP : Tiny_Op<"print">  {
  let summary =  "print operation";
  let description = [{
      The print builtin operation prints a given input tensor
  }];

  let arguments = (ins I32Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";

  let printer = [{ return ::print(printer,  *this); }];
  let parser  = [{ return ::parser$cppClass(paser, result); }];
}
```

至此我们完成了 Tiny Dialect 的定义，并且定义了一个用于打印字符串的 `Print` 算子，其对应的 Tiny IR 如下所示：

```text
tiny.print %0 : tensor<6xi32>
```

接下来我们要编写一个函数完成 Tiny AST 到 Tiny IR 的转换工作，具体的可以参考 [MLIRGen.cpp](./mlir/MLIRGen.cpp) 这个文件。
最后，您可以使用命令 `tinyc hello.tiny -emit=mlir` 来查看完整的 Tiny IR：

``` text
$echo 'def main() {print("Hello World!");}' | ./bin/tinyc -emit=mlir
builtin.module  {
  builtin.func @main() {
    %0 = tiny.constant dense<[72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 92, 110]> : tensor<14xi32>
    tiny.print %0 : tensor<14xi32>
    tiny.return
  }
}

```

## Lowering MLIR to Lower-Level Dialect

按照 MLIR 的设计，我们最终需要输出 LLVM IR 才可以通过 LLVM Codegen 生成不同的机器码并执行, 这中间的转换工作是通过实现不同的 Pass 完成的，
这些 Pass 中有些负责对 IR 的分析优化，有些负责将一个 Dialect 转换成另一个 Dialect.

MLIR 中提供了 [Converter Framework](https://mlir.llvm.org/getting_started/Glossary/#conversion) 完成这类转换工作，一个完备的 Converter
至少提供两个信息：

1. Converter Target, Converter 的目标描述，需要配置 illegal 和 legal 信息。
1. Converter Pattern, 具体的 Converter 实现。

### Converter Target

``` cpp
void TinyToAffineLoweringPass::runOnFunction() {
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::AffineDialect, mlir::memref::MemRefDialect,
                         mlir::StandardOpsDialect>();

  target.addIllegalDialect<TinyDialect>();
  target.addLegalOp<PrintOp>();
  ...
}
```

在上述代码中，我们使用 `target.addLegalDialect` 设置合法的 Dialect，`target.addIllegalDialect` 设置非法的 Dialect，这里将
`TinyDialect` 设置为非法，意味着 TinyDialect 将不会出现在 AffineLoweringPass 之后。

值得注意的是，我们可以使用 `target.addLegalOp<PrintOp>()` 将 `PrintOp` 设置为例外，同时我们也需要更新下 `PrintOp` 的输入定义，增加
`I32MemRef` 类型：

``` cpp
let arguments = (ins AnyTypeOf<[I32Tensor, I32MemRef]>:$input);
```

### Converter Pattern

当我们定义好 Converter 的目标后，接下来就是如何具体地将一个 Illegal Operation 转换成一个 Legal Operation, 以下面 `tiny.return` 的 lowering 代码为例,
`ReturnOpLowering` 将会把 `tiny.return` 重写为 StandardOpsDialect 中的 `std.return` Operation.

``` cpp
struct ReturnOpLowering : public OpRewritePattern<tiny::ReturnOp> {
  using OpRewritePattern<tiny::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tiny::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};
```

### 运行式例

``` bash
$ echo 'def main() {print("Hello World!");}' | ./bin/tinyc -emit=mlir-affine
builtin.module  {
  builtin.func @main() {
    %c33_i32 = constant 33 : i32
    %c100_i32 = constant 100 : i32
    %c114_i32 = constant 114 : i32
    %c87_i32 = constant 87 : i32
    %c32_i32 = constant 32 : i32
    %c111_i32 = constant 111 : i32
    %c108_i32 = constant 108 : i32
    %c101_i32 = constant 101 : i32
    %c72_i32 = constant 72 : i32
    %0 = memref.alloc() : memref<12xi32>
    affine.store %c72_i32, %0[0] : memref<12xi32>
    affine.store %c101_i32, %0[1] : memref<12xi32>
    affine.store %c108_i32, %0[2] : memref<12xi32>
    affine.store %c108_i32, %0[3] : memref<12xi32>
    affine.store %c111_i32, %0[4] : memref<12xi32>
    affine.store %c32_i32, %0[5] : memref<12xi32>
    affine.store %c87_i32, %0[6] : memref<12xi32>
    affine.store %c111_i32, %0[7] : memref<12xi32>
    affine.store %c114_i32, %0[8] : memref<12xi32>
    affine.store %c108_i32, %0[9] : memref<12xi32>
    affine.store %c100_i32, %0[10] : memref<12xi32>
    affine.store %c33_i32, %0[11] : memref<12xi32>
    tiny.print %0 : memref<12xi32>
    memref.dealloc %0 : memref<12xi32>
    return
  }
}
```

## Lowering to LLVM IR and Setup JIT

至此，我们已经将 `TinyDialect` lower 到了一些更低层级的 Dialect: `AffineDialect`, `MemRefDialect` 和 `StandardOpsDialect`。
在这一节中，我们会将这些 Dialect 全部 lower 到 LLVM IR，从而可以调用 LLVM CodeGen 生成汇编程序，并且可以启动 JIT 尝试运行它。

需要注意到是，我们在 [Lowering MLIR to Lower-Level Dialect](#lowering-mlir-to-lower-level-dialect) 中，并没有将 `tiny.print` operation
进行 lower，我们要在 LLVM Converter 中对 `tiny.print` 进行转换：

``` cpp
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }
```

参考 [上一节](#lowering-mlir-to-lower-level-dialect) 来配置对应 Converter Target:

``` cpp
mlir::ConversionTarget target(getContext());
target.addLegalDialect<mlir::LLVMDialect>();
target.addLegalOp<mlir::ModuleOp>();
...
```

和 Converter Pattern：

``` cpp
...
LLVMTypeConverter typeConverter(&getContext());
mlir::RewritePatternSet patterns(&getContext());
mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

// The only remaining operation, to lower from the `toy` dialect, is the
// PrintOp.
patterns.add<PrintOpLowering>(&getContext());
...
```

### Setting up a JIT

MLIR 提供了 Execution Engine 来执行 LLVM Dialect:

``` cpp
...
// An optimization pipeline to use within the execution engine.
auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

// Create an MLIR execution engine. The execution engine eagerly JIT-compiles
// the module.
auto maybeEngine = mlir::ExecutionEngine::create(
  module, /*llvmModuleBuilder=*/nullptr, optPipeline);
assert(maybeEngine && "failed to construct an execution engine");
auto &engine = maybeEngine.get();

// Invoke the JIT-compiled function.
auto invocationResult = engine->invokePacked("main");
...
```

完整代码可以参考 [main.cpp](./tinyc/main.cpp) 中的 runJIT 函数。

接下来可以使用如下命令执行 Tiny 程序：

``` bash
$ echo 'def main() {print("Hello World!");}' | ./bin/tinyc -emit=jit
Hello World!
```
