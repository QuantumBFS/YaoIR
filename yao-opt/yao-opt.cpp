//===- yao-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Yao/YaoDialect.h"
#include "Yao/YaoOps.h"

int main(int argc, char **argv) {
  using namespace mlir;
  mlir::registerAllPasses();
  // TODO: Register yao passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::yao::YaoDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  MLIRContext context;
  context.disableMultithreading();
  context.getOrLoadDialect<mlir::yao::YaoDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();
  // MLIRContext context;

  mlir::OpBuilder builder(&context);
  std::vector<mlir::Type> rettypes;
  auto funcType = builder.getFunctionType(rettypes, rettypes);
  mlir::FuncOp function = mlir::FuncOp(
      mlir::FuncOp::create(builder.getUnknownLoc(), "moo", funcType));
      
//   auto mod =
//       mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc());

  auto entryBlock = function.addEntryBlock();

  builder.setInsertionPointToStart(entryBlock);
  // auto H = builder.create<mlir::yao::HOp>(builder.getUnknownLoc(), mlir::yao::GateType::get(&context, 1));
  auto i1Ty = builder.getIndexType();
  auto vfalse = builder.create<mlir::ConstantOp>(
      builder.getUnknownLoc(), i1Ty, builder.getIntegerAttr(i1Ty, 0));
  std::vector<mlir::Value> ops = {vfalse, vfalse};
  auto locations = builder.create<mlir::yao::LocationsOp>(builder.getUnknownLoc(), mlir::yao::LocationsType::get(&context, 1), ops);
  std::vector<mlir::Value> locs = {locations};
  llvm::errs() << (mlir::Value)builder.create<mlir::yao::MeasureOp>(builder.getUnknownLoc(), mlir::yao::MeasureResultType::get(&context, 1), locs) << "\n";

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Yao optimizer driver\n", registry));
}
