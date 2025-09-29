# Number of Floating Point operations 
This repository provides an LLVM compiler pass that instruments CUDA kernels to count floating-point binary operations at runtime.  
The pass detects operations such as addition, subtraction, multiplication, division, remainder, and fused multiply-add (FMA), and inserts calls to a runtime hook function to record their occurrence.


## Build the pass 
```
mkdir build && cd build
```

```
cmake -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm ..
```
```
make
```

## Compile and Run

```
# Emit device LLVM IR (device-only)
clang++ -x cuda -c ./test/device.cu --cuda-path=/usr/local/cuda \
 --cuda-gpu-arch=sm_90 --cuda-device-only -O1 -emit-llvm -S -o device.ll

# Insert the pass (pipeline name: insert-record-event)
opt -load-pass-plugin=build/InsertRecordEventPass.so --passes=insert-record-event -S device.ll -o device_inst.ll

# Lower to PTX
llc -march=nvptx64 -mcpu=sm_90 device_inst.ll -o device_inst.ptx

# PTX -> cubin
nvcc -arch=sm_90 -cubin device_inst.ptx -o device_inst.cubin

# Device link (for any device symbols)
nvcc -arch=sm_90 -dlink device_inst.cubin -o device_dlink.o

# Build host
nvcc -arch=sm_90 -c ./test/host.cu -o host.o

# Link (-lcudart for Runtime API | -lcuda for Driver API), I am writing driver api codes because of the flexibility with the context management, which i need for llvm instrumentation
nvcc -arch=sm_90 host.o device_dlink.o -lcuda -o main

./main
```
