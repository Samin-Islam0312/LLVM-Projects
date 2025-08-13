#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"

#include "llvm/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;   // Identify the kernel in the program file
static bool isCudaKernel(Function &F) {
  // CUDA kernels are marked in nvvm.annotations: ->>>> !nvvm.annotations = !{ !{ptr @kernel, !"kernel", i32 1}, ... }
  Module *M = F.getParent();
  NamedMDNode *Anno = M->getNamedMetadata("nvvm.annotations");
  if (!Anno) return false;
  for (auto *Op : Anno->operands()) {
    if (auto *N = dyn_cast<MDNode>(Op)) {
      if (N->getNumOperands() < 2) continue;
      if (auto *V = dyn_cast_or_null<ValueAsMetadata>(N->getOperand(0))) {
        if (V->getValue() == &F) {
          if (auto *S = dyn_cast_or_null<MDString>(N->getOperand(1)))
            if (S->getString() == "kernel")
              return true;
        }
      }
    }
  }
  return false;
}

namespace {
struct InsertRecordEventPass : PassInfoMixin<InsertRecordEventPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    LLVMContext &Ctx = M.getContext();

    // void __record_binop(i32 id)
    FunctionCallee Hook =
        M.getOrInsertFunction("__record_binop",
          FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false));
    cast<Function>(Hook.getCallee())->setCallingConv(CallingConv::C);

    auto isNegConstFP = [](Value *V) -> bool {
      if (auto *CF = dyn_cast<ConstantFP>(V))
        return CF->isNegative() && !CF->isZero(); // true for -K (K>0)
      return false;
    };

    auto isFNeg = [](Value *V) -> bool {
      // fneg x
      if (auto *U = dyn_cast<UnaryOperator>(V))
        return U->getOpcode() == Instruction::FNeg;
      // canonical form: fsub -0.0, x
      if (auto *BO = dyn_cast<BinaryOperator>(V)) {
        if (BO->getOpcode() == Instruction::FSub)
          if (auto *L = dyn_cast<ConstantFP>(BO->getOperand(0)))
            return L->isNegative() && L->isZero();
      }
      return false;
    };

    auto opcodeId = [](unsigned Opc) -> int {
      switch (Opc) {
        case Instruction::FAdd: return 0;
        case Instruction::FSub: return 1;
        case Instruction::FMul: return 2;
        case Instruction::FDiv: return 3;
        case Instruction::FRem: return 4;
        default: return -1;
      }
    };

    bool Changed = false;

    for (Function &F : M) {
      if (F.isDeclaration()) continue;
      if (!isCudaKernel(F))  continue;  

      SmallVector<Instruction*, 128> Sites;
      SmallVector<int, 128>          Codes;

      for (auto &BB : F) {
        for (auto &I : BB) {
          // --- Count FMA (llvm intrinsic or libdevice) as id=5 ---
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            if (Function *Callee = CB->getCalledFunction()) {
              bool isFMA = (Callee->getIntrinsicID() == Intrinsic::fma);
              if (!isFMA) {
                StringRef N = Callee->getName();
                // libdevice symbols frequently appear this way
                isFMA = N.starts_with("__nv_fma") || N.starts_with("__nv_fmaf");
              }
              if (isFMA && I.getType()->isFPOrFPVectorTy()) {
                Sites.push_back(&I);
                Codes.push_back(5);
                continue; // do not also try to classify as add/mul
              }
            }
          }

          // Count FP binary ops
          if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
            if (!BO->getType()->isFPOrFPVectorTy()) continue;

            unsigned opc = BO->getOpcode();
            if (opc == Instruction::FAdd) {
              // a negated operand as semantic subtraction
              Value *L = BO->getOperand(0);
              Value *R = BO->getOperand(1);
              bool looksSub = isNegConstFP(L) || isNegConstFP(R) || isFNeg(L) || isFNeg(R);
              Sites.push_back(&I);
              Codes.push_back(looksSub ? 1 : 0);
              continue;
            }

            int id = opcodeId(opc);
            if (id >= 0) {
              Sites.push_back(&I);
              Codes.push_back(id);
            }
          }
        }
      }

      if (Sites.empty()) continue;

      for (size_t i = 0; i < Sites.size(); ++i) {
        Instruction *Inst = Sites[i];
        int id = Codes[i];
        IRBuilder<> B(Inst->getParent(), std::next(Inst->getIterator()));
        B.CreateCall(Hook, { B.getInt32(id) });
      }

      Changed = true;
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

}
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  using namespace llvm;
  return {
    LLVM_PLUGIN_API_VERSION,
    "InsertRecordEventPass",
    "1.0",
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, ModulePassManager &MPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "insert-record-event") {
            MPM.addPass(InsertRecordEventPass());
            return true;
          }
          return false;
        });
    }
  };
}
