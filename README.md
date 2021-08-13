# MLIR--Hello World

最近的一些工作会接触到 AI 编译器相关的技术，其中 MLIR 是 Google 在 2019 年的 GCP 上提出的一种新的 AI 编译基础设施,
其目的是希望现有的编译优化技术（例如 XLA、TVM等）能够以 Dialect 的方式接入进来，从而使各家的编译优化技术能够一起发挥效用。

和大部分程序语言的 101 一样，本文尝试实现一种新的语言 **Tiny**, 实现了打印 `Hello World!` 的功能。

## PreRequirements


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
$ tiny hello.tiny -emit ast
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

因为目前 MLIR 还没有一条完备的通路可以支持 String 类型，所以在 AST 这里用 Int Tensor 来处理。

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

至此我们完成了 Tiny Dialect 的定义，并且定义了一个用于打印字符串的 `Print` 算子，这个算子对应的 Tiny IR 如下所示：

```text
tiny.print %0 : tensor<6xi32>
```

接下来我们要编写一个函数完成 Tiny AST 到 Tiny IR 的转换工作，具体的可以参考 [MLIRGen.cpp](./mlir/MLIRGen.cpp) 这个文件。
最后，您可以使用命令 `tinyc hello.tiny -emit=mlir` 来查看完整的 Tiny IR：

``` text
$./bin/tinyc hello.tiny -emit=mlir
builtin.module  {
  builtin.func @main() {
    %0 = tiny.constant dense<[72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33, 92, 110]> : tensor<14xi32>
    tiny.print %0 : tensor<14xi32>
    tiny.return
  }
}

```

## Lowering to LLVM IR


