# Number of Floating Point operations 
This repository provides an LLVM compiler pass that instruments CUDA kernels to count floating-point binary operations at runtime.  
The pass detects operations such as addition, subtraction, multiplication, division, remainder, and fused multiply-add (FMA), and inserts calls to a runtime hook function to record their occurrence.
