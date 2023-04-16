#pragma once

#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

struct Tensor{
    float* buf_cpu = nullptr;
    float* buf_gpu = nullptr;
    float* buf     = nullptr;

    size_t ndim = 0;
    size_t shape[4];

    // alloc
    Tensor(std::vector<int> shape_){
        ndim = shape_.size();
        for(size_t i=0; i<ndim; ++i){
            shape[i] = shape_[i];
        }

        size_t n = num_elem();
        buf_cpu = (float *) malloc(n * sizeof(float));
        buf = buf_cpu;
    }

    // copy
    Tensor(std::vector<int> shape_, float *buf_){
        ndim = shape_.size();
        for (size_t i=0; i<ndim; ++i){
            shape[i] = shape_[i];
        }

        size_t n = num_elem();
        buf_cpu = (float *)malloc(n * sizeof(float));
        memcpy(buf_cpu, buf_, n*sizeof(float));
        buf = buf_cpu;
    }


    // desctruct
    ~Tensor(){
        if (buf_cpu != nullptr) free(buf_cpu);
        if (buf_gpu != nullptr) free(buf_cpu);
        // if (buf != nullptr) free(buf);
    }

    void set_zero(){
        size_t n = num_elem();
        for (size_t i=0; i<n; ++i){
            buf[i] = 0.0;
        }
    }

    size_t num_elem(){
        size_t sz = 1;
        for(size_t i=0; i<ndim; ++i){
            sz *= shape[i];
        }
        return sz;
    }

  void gpu(){
    if (is_offloading()) return;
    // allocate device memory 
    cudaMalloc(&buf_gpu, sizeof(float)*num_elem());

    // copy data from host into device memory
    cudaMemcpy(
        buf_gpu, buf_cpu, // gpu from cpu
        sizeof(float)*num_elem(), 
        cudaMemcpyHostToDevice
    );
    
    // destruct host memory
    free(buf_cpu); // 'buf' freed too !
    buf = buf_gpu; // change pointer direction
  }

  void cpu(){
    if (is_offloading()==false) return; // already in cpu memory
    // allocate host memory
    buf_cpu = (float *)malloc(sizeof(float)*num_elem());

    // copy data from device into host memory
    cudaMemcpy(
        buf_cpu, buf_gpu, // cpu from gpu
        sizeof(float)*num_elem(), 
        cudaMemcpyDeviceToHost
    );

    free(buf_gpu); // destruct device memory
    // free(buf);
    // change buf direction
    buf = buf_cpu;
  }

  bool is_offloading(){
    if (buf==buf_gpu) return true;
    else              return false;
  }

};



/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

