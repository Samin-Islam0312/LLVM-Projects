extern "C" {
__device__ unsigned long long gBinOpCounts[6];  // 0..5

// 0=FAdd, 1=FSub, 2=FMul, 3=FDiv, 4=FRem, 5=FMA 
__device__ void __record_binop(int id) {
  if (id < 0 || id > 5) return;
  atomicAdd(&gBinOpCounts[id], 1ULL);
}

extern "C" __global__
void divKernel(const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float x = A[i], y = B[i];

    // 1 × FMA
    float r = fmaf(x, y, 1.0f);

    // 3 × add
    r = r + 1.0f;
    r = r + 2.0f;
    r = r + 3.0f;

    // 8 × sub
    r = r - 1.0f; r = r - 2.0f; r = r - 3.0f; r = r - 4.0f;
    r = r - 5.0f; r = r - 6.0f; r = r - 7.0f; r = r - 8.0f;

    C[i] = r / (y + 1e-20f);
  }
}

} // extern "C"
