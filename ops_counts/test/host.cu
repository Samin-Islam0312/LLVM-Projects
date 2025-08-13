// file: host_driver.cpp
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

static void ck(CUresult r, const char* where) {
  if (r != CUDA_SUCCESS) {
    const char *name=nullptr, *str=nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &str);
    std::fprintf(stderr, "CUDA Driver error at %s: %s (%s)\n",
                 where, name?name:"?", str?str:"?");
    std::exit(1);
  }
}

static std::string load_text_file(const char* path) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) { std::perror(path); std::exit(1); }
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <instrumented.ptx> [N=1048576] [block=256]\n", argv[0]);
    return 1;
  }
  const char* ptxPath = argv[1];
  int N = (argc >= 3) ? std::atoi(argv[2]) : (1<<20);
  int block = (argc >= 4) ? std::atoi(argv[3]) : 256;
  if (N <= 0) N = 1<<20;
  if (block <= 0) block = 256;
  int grid = (N + block - 1) / block;

  ck(cuInit(0), "cuInit");
  CUdevice dev; ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx; ck(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

  // Load instrumented PTX
  std::string ptx = load_text_file(ptxPath);
  CUmodule mod; ck(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

  // Resolve kernel and globals (names must match your device file)
  CUfunction k; ck(cuModuleGetFunction(&k, mod, "divKernel"), "cuModuleGetFunction(divKernel)");

  // Per-op counters: gBinOpCounts[6] (u64)
  CUdeviceptr dCounts; size_t countsBytes = 0;
  ck(cuModuleGetGlobal(&dCounts, &countsBytes, mod, "gBinOpCounts"), "cuModuleGetGlobal(gBinOpCounts)");
  if (countsBytes < 6 * sizeof(unsigned long long)) {
    std::fprintf(stderr, "gBinOpCounts size mismatch: %zu\n", countsBytes);
    return 1;
  }
  ck(cuMemsetD8(dCounts, 0, 6*sizeof(unsigned long long)), "reset gBinOpCounts");

  // Buffers
  size_t bytes = static_cast<size_t>(N) * sizeof(float);
  CUdeviceptr dA, dB, dC;
  ck(cuMemAlloc(&dA, bytes), "cuMemAlloc(A)");
  ck(cuMemAlloc(&dB, bytes), "cuMemAlloc(B)");
  ck(cuMemAlloc(&dC, bytes), "cuMemAlloc(C)");

  std::vector<float> hA(N), hB(N);
  for (int i=0;i<N;i++){ hA[i]=float(i+1); hB[i]=float((i%7)+1); }
  ck(cuMemcpyHtoD(dA, hA.data(), bytes), "cuMemcpyHtoD(A)");
  ck(cuMemcpyHtoD(dB, hB.data(), bytes), "cuMemcpyHtoD(B)");

  // Launch
  void* params[] = { &dA, &dB, &dC, &N };
  ck(cuLaunchKernel(k,
                    grid, 1, 1,
                    block, 1, 1,
                    0, 0, params, nullptr),
     "cuLaunchKernel");
  ck(cuCtxSynchronize(), "cuCtxSynchronize");

  // Read back per-op counts
  unsigned long long hCounts[6] = {};
  ck(cuMemcpyDtoH(hCounts, dCounts, 6*sizeof(unsigned long long)), "cuMemcpyDtoH(gBinOpCounts)");

  const char* names[6] = {"FAdd","FSub","FMul","FDiv","FRem","FMA"};
  std::puts("Per-opcode FP binary counts:");
  for (int i=0;i<6;i++)
    std::printf("  %-4s = %llu\n", names[i], (unsigned long long)hCounts[i]);

  // (Optional) pull result to verify math path executed
  std::vector<float> hC; hC.resize(N);
  ck(cuMemcpyDtoH(hC.data(), dC, bytes), "cuMemcpyDtoH(C)");

  cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
