; ModuleID = 'device.ll'
source_filename = "./test/device.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@gBinOpCounts = dso_local addrspace(1) externally_initialized global [6 x i64] zeroinitializer, align 8
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @gBinOpCounts to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define dso_local void @__record_binop(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ugt i32 %0, 5
  br i1 %2, label %7, label %3

3:                                                ; preds = %1
  %4 = zext nneg i32 %0 to i64
  %5 = getelementptr inbounds [6 x i64], ptr addrspacecast (ptr addrspace(1) @gBinOpCounts to ptr), i64 0, i64 %4
  %6 = atomicrmw add ptr %5, i64 1 seq_cst, align 8
  br label %7

7:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @divKernel(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = tail call noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = tail call noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %7 = mul i32 %5, %6
  %8 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %9 = add i32 %7, %8
  %10 = icmp slt i32 %9, %3
  br i1 %10, label %11, label %32

11:                                               ; preds = %4
  %12 = sext i32 %9 to i64
  %13 = getelementptr inbounds float, ptr %0, i64 %12
  %14 = load float, ptr %13, align 4, !tbaa !8
  %15 = getelementptr inbounds float, ptr %1, i64 %12
  %16 = load float, ptr %15, align 4, !tbaa !8
  %17 = tail call noundef float @llvm.fma.f32(float %14, float %16, float 1.000000e+00)
  call void @__record_binop(i32 5)
  %18 = fadd contract float %17, 1.000000e+00
  call void @__record_binop(i32 0)
  %19 = fadd contract float %18, 2.000000e+00
  call void @__record_binop(i32 0)
  %20 = fadd contract float %19, 3.000000e+00
  call void @__record_binop(i32 0)
  %21 = fadd contract float %20, -1.000000e+00
  call void @__record_binop(i32 1)
  %22 = fadd contract float %21, -2.000000e+00
  call void @__record_binop(i32 1)
  %23 = fadd contract float %22, -3.000000e+00
  call void @__record_binop(i32 1)
  %24 = fadd contract float %23, -4.000000e+00
  call void @__record_binop(i32 1)
  %25 = fadd contract float %24, -5.000000e+00
  call void @__record_binop(i32 1)
  %26 = fadd contract float %25, -6.000000e+00
  call void @__record_binop(i32 1)
  %27 = fadd contract float %26, -7.000000e+00
  call void @__record_binop(i32 1)
  %28 = fadd contract float %27, -8.000000e+00
  call void @__record_binop(i32 1)
  %29 = fadd contract float %16, 0x3BC79CA100000000
  call void @__record_binop(i32 0)
  %30 = fdiv contract float %28, %29
  call void @__record_binop(i32 3)
  %31 = getelementptr inbounds float, ptr %2, i64 %12
  store float %30, ptr %31, align 4, !tbaa !8
  br label %32

32:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_90" "target-features"="+ptx85,+sm_90" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_90" "target-features"="+ptx85,+sm_90" "uniform-work-group-size"="true" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4}
!llvm.ident = !{!5, !6}
!nvvmir.version = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @divKernel, !"kernel", i32 1}
!5 = !{!"Ubuntu clang version 19.1.7 (++20250114103320+cd708029e0b2-1~exp1~20250114103432.75)"}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = !{i32 2, i32 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
