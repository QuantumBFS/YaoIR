//===- YaoOps.cpp - Yao dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Yao/YaoOps.h"
#include "Yao/YaoDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#define GET_OP_CLASSES
#include "Yao/YaoOps.cpp.inc"

#include "mlir/IR/PatternMatch.h"

using namespace mlir::yao;
using namespace llvm;
using namespace mlir;


struct ChainRemoveIdentity : public mlir::OpRewritePattern<ChainOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ChainRemoveIdentity(mlir::MLIRContext *context)
      : OpRewritePattern<ChainOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ChainOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    SmallVector<mlir::Value, 2> vals;
    bool seenIdentity = false;
    for (auto op : op.getOperands()) {
        if (op.getDefiningOp<yao::IOp>()) {
            seenIdentity = true;
        } else {
            vals.push_back(op);
        }
    }

    if (!seenIdentity) return failure();

    if (vals.size() == 0) {
        rewriter.replaceOpWithNewOp<IOp>(op, op.getType());
        return success();
    }
    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOpWithNewOp<ChainOp>(op, op.getType(), vals);
    return success();
  }
};


struct Unchained : public mlir::OpRewritePattern<ChainOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  Unchained(mlir::MLIRContext *context)
      : OpRewritePattern<ChainOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ChainOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1) return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, op.getOperand(0));
    return success();
  }
};

struct FindIdentity : public mlir::OpRewritePattern<ChainOp> {
    FindIdentity(mlir::MLIRContext *context)
        : OpRewritePattern<ChainOp>(context, 1) {}

    mlir::LogicalResult
    matchAndRewrite(ChainOp src,
                  mlir::PatternRewriter &rewriter) const override {
        auto operands = src.getOperands();
        auto prev = src.getOperands().begin();
        SmallVector<Value, 4> args;
        for(auto curr = std::next(operands.begin()); curr!=operands.end(); ++curr) {
            if (curr != prev) {
                args.push_back(*prev);
            }
            prev = curr;
        }
        rewriter.replaceOpWithNewOp<ChainOp>(src, src.getType(), args);
        return success();
    }
};

void ChainOp::getCanonicalizationPatterns(
     OwningRewritePatternList &results, MLIRContext *context) {
   results.insert<ChainRemoveIdentity, Unchained, FindIdentity>(context);
 }


 struct ReChainer : public mlir::OpRewritePattern<GateOp> {
   /// We register this pattern to match every toy.transpose in the IR.
   /// The "benefit" is used by the framework to order the patterns and process
   /// them in order of profitability.
   ReChainer(mlir::MLIRContext *context)
       : OpRewritePattern<GateOp>(context, /*benefit=*/1) {}

   /// This method is attempting to match a pattern and rewrite it. The rewriter
   /// argument is the orchestrator of the sequence of rewrites. It is expected
   /// to interact with it to perform any changes to the IR from here.
   mlir::LogicalResult
   matchAndRewrite(GateOp orig,
                   mlir::PatternRewriter &rewriter) const override {

    SmallVector<GateOp, 4> ops;
    for (auto op = orig.getOperation(); ; op = op->getPrevNode()) {
        if (auto gop = dyn_cast<GateOp>(op)) {
            if (gop.locations() == orig.locations()) {
                ops.push_back(gop);
            //} else if (/* known not to be*/) {
            //    continue
            } else {
                break;
            }
        }
        if (&op->getBlock()->front() == op) break;
    }

    if (ops.size() == 1)
        return failure();
    
     // Otherwise, we have a redundant transpose. Use the rewriter.
     SmallVector<Value, 4> arg;
     for(auto gop : ops) {
       arg.push_back(gop.QuOp());
     }
     std::reverse(arg.begin(), arg.end());
     auto chain = rewriter.create<ChainOp>(orig.getLoc(), arg[0].getType(), arg);
     rewriter.replaceOpWithNewOp<GateOp>(orig, chain, orig.locations());
     ops.erase(ops.begin());
     for(auto gop : ops) {
       rewriter.eraseOp(gop);
     }
     return success();
   }
 };

void GateOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReChainer>(context);
}
