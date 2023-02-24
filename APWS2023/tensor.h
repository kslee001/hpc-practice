#pragma once

#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// You can modify the data structure as you want
struct Tensor {
  // Alloc memory
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (int i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    int n = num_elem();

    // cpu
    buf = (float *) malloc(n * sizeof(float));

    // gpu
    cudaMalloc(
      &buf_gpu,
      sizeof(float)*n
    );

  }

  // Alloc memory and copy
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (int i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    int n = num_elem();

    // cpu
    buf = (float *) malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) { buf[i] = buf_[i]; }

    // gpu
    cudaMalloc(
      &buf_gpu,
      sizeof(float)*n
    );
    cudaMemcpy(
      buf_gpu,
      buf_,
      sizeof(float)*n,
      cudaMemcpyHostToDevice
    );
  }

  ~Tensor() {
    if (buf != nullptr) free(buf);
    if (buf_gpu != nullptr) cudaFree(buf_gpu);
  }

  int num_elem() {
    int sz = 1;
    for (int i = 0; i < ndim; i++) sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf = nullptr;
  float *buf_gpu = nullptr;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  int ndim = 0;
  int shape[4];
};
