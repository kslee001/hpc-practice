#include <stdlib.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tensor.h"
#include "uNet.h"
#include "util.h"



  // Parameters for U-Net
  Tensor *inc_double_conv_0_weight;
  Tensor *inc_double_conv_1_weight;
  Tensor *inc_double_conv_1_bias;
  Tensor *inc_double_conv_3_weight;
  Tensor *inc_double_conv_4_weight;
  Tensor *inc_double_conv_4_bias;
  Tensor *down1_maxpool_conv_1_double_conv_0_weight;
  Tensor *down1_maxpool_conv_1_double_conv_1_weight;
  Tensor *down1_maxpool_conv_1_double_conv_1_bias;
  Tensor *down1_maxpool_conv_1_double_conv_3_weight;
  Tensor *down1_maxpool_conv_1_double_conv_4_weight;
  Tensor *down1_maxpool_conv_1_double_conv_4_bias;
  Tensor *down2_maxpool_conv_1_double_conv_0_weight;
  Tensor *down2_maxpool_conv_1_double_conv_1_weight;
  Tensor *down2_maxpool_conv_1_double_conv_1_bias;
  Tensor *down2_maxpool_conv_1_double_conv_3_weight;
  Tensor *down2_maxpool_conv_1_double_conv_4_weight;
  Tensor *down2_maxpool_conv_1_double_conv_4_bias;
  Tensor *up1_up_weight;
  Tensor *up1_up_bias;
  Tensor *up1_conv_double_conv_0_weight;
  Tensor *up1_conv_double_conv_1_weight;
  Tensor *up1_conv_double_conv_1_bias;
  Tensor *up1_conv_double_conv_3_weight;
  Tensor *up1_conv_double_conv_4_weight;
  Tensor *up1_conv_double_conv_4_bias;
  Tensor *up2_up_weight;
  Tensor *up2_up_bias;
  Tensor *up2_conv_double_conv_0_weight;
  Tensor *up2_conv_double_conv_1_weight;
  Tensor *up2_conv_double_conv_1_bias;
  Tensor *up2_conv_double_conv_3_weight;
  Tensor *up2_conv_double_conv_4_weight;
  Tensor *up2_conv_double_conv_4_bias;
  Tensor *outc_conv_weight;
  Tensor *outc_conv_bias;
  Tensor *inc_batchnorm_0_running_mean;
  Tensor *inc_batchnorm_0_running_var;
  Tensor *down1_batchnorm_0_running_mean;
  Tensor *down1_batchnorm_0_running_var;
  Tensor *down2_batchnorm_0_running_mean;
  Tensor *down2_batchnorm_0_running_var;
  Tensor *up1_batchnorm_0_running_mean;
  Tensor *up1_batchnorm_0_running_var;
  Tensor *up2_batchnorm_0_running_mean;
  Tensor *up2_batchnorm_0_running_var;
  Tensor *inc_batchnorm_1_running_mean;
  Tensor *inc_batchnorm_1_running_var;
  Tensor *down1_batchnorm_1_running_mean;
  Tensor *down1_batchnorm_1_running_var;
  Tensor *down2_batchnorm_1_running_mean;
  Tensor *down2_batchnorm_1_running_var;
  Tensor *up1_batchnorm_1_running_mean;
  Tensor *up1_batchnorm_1_running_var;
  Tensor *up2_batchnorm_1_running_mean;
  Tensor *up2_batchnorm_1_running_var;

  // intermediate features
  Tensor *inc_conv_0_output;
  Tensor *inc_batchnorm_0_output;
  Tensor *inc_conv_1_output;
  Tensor *inc_batchnorm_1_output;
  Tensor *down1_maxpool2d_0_output;
  Tensor *down1_conv_0_output;
  Tensor *down1_batchnorm_0_output;
  Tensor *down1_conv_1_output;
  Tensor *down1_batchnorm_1_output;
  Tensor *down2_maxpool2d_0_output;
  Tensor *down2_conv_0_output;
  Tensor *down2_batchnorm_0_output;
  Tensor *down2_conv_1_output;
  Tensor *down2_batchnorm_1_output;
  Tensor *up1_convt_0_output;
  Tensor *up1_concat_0_output;
  Tensor *up1_conv_0_output;
  Tensor *up1_batchnorm_0_output;
  Tensor *up1_conv_1_output;
  Tensor *up1_batchnorm_1_output;
  Tensor *up2_convt_0_output;
  Tensor *up2_concat_0_output;
  Tensor *up2_conv_0_output;
  Tensor *up2_batchnorm_0_output;
  Tensor *up2_conv_1_output;
  Tensor *up2_batchnorm_1_output;
  Tensor *outc_conv_0_output;



/* DEFINE */
#define NUM_GPU 4
#define BATCH_SIZE 1


#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


/* BLOCK & GRIDS */
#define BLOCK_SIZE 32
dim3 grid_gen(Tensor* curTensor, int bs){
  return dim3( (curTensor->shape[0]+bs-1)/bs, (curTensor->shape[1]+bs-1)/bs  );
}


/* UTILS */
void debug(Tensor* cur, const char *msg){
  float *t = (float*) malloc( cur->num_elem() * sizeof(float));
  cudaMemcpy(t, cur->buf_gpu, cur->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < cur->num_elem(); i++) {
    if (fabs(t[i] - cur->buf[i]) > 1e-6 ) {
      printf("%s, Difference at %d: GPU %f, CPU %f\n", msg, i, t[i], cur->buf[i]);
      exit(0);
    }
  }
}

void print(Tensor* cur){
  float *t = (float*) malloc( cur->num_elem() * sizeof(float));
  cudaMemcpy(t, cur->buf_gpu, cur->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);
  printf("\n-- debug --\n");
  for(int i=0; i<5; ++i){
    printf("%f ", t[i]);
  }  
  printf("\n");
}



/* KERNELS */
__global__ void Conv2dKernel(
  float *input, float* weight, float* bias, float* output, 
  int N, int C, int H, int W,
  int K, int R, int S, 
  int ON, int OC, int OH, int OW,
  int stride, int pad, int dilation, bool has_bias
){
  const int ow = blockDim.x * blockIdx.x + threadIdx.x;
  const int oh = blockDim.y * blockIdx.y + threadIdx.y;
  const int oc = blockIdx.z;

  if (ow >= OW || oh >= OH || oc >= OC) return;

  float sum = has_bias ? bias[oc] : 0;

  for(int n=0; n<N; ++n){
    for(int c=0; c<C; ++c){
      for(int r=0; r<R; ++r){
        for(int s=0; s<S; ++s){
          const int h= oh*stride - pad + r*dilation;
          const int w= ow*stride - pad + s*dilation;
          if(h<0||h>=H||w<0||w>=W) continue;
          sum += input[((n*C+c)*H+h)*W+w]*
          weight[((oc*C+c)*R+r)*S+s];
        }
      }
    }
  }
  output[((oc*OH+oh)*OW)+ow] = sum;
}

void Conv2dDevice(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
                             int stride, int pad, int dilation, bool has_bias){
  int N=input->shape[0], C=input->shape[1], H=input->shape[2], W=input->shape[3];
  int K=weight->shape[0], R=weight->shape[2], S=weight->shape[3];
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2*pad - (((R-1)*dilation) + 1))/stride;
  const int OW = 1 + (W + 2*pad - (((S-1)*dilation) + 1))/stride;
  float* b = has_bias ? bias->buf_gpu : nullptr;

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((OW+BLOCK_SIZE-1)/BLOCK_SIZE, (OH+BLOCK_SIZE-1)/BLOCK_SIZE, K);

  Conv2dKernel<<<gridDim, blockDim>>>(
    input->buf_gpu, weight->buf_gpu, b, output->buf_gpu, 
    N, C, H, W, K, R, S, ON, OC, OH, OW, stride, pad, dilation, has_bias
  ); 
  cudaDeviceSynchronize();
}



__global__ void ReLUKernel(float *inout, int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N*C*H*W) return;
  int w = idx % W;
  idx /= W;
  int h = idx % H;
  idx /= H;
  int c = idx % C;
  idx /= C;
  int n = idx;
  int offset = ((n * C + c) * H + h) * W + w;
  inout[offset] = inout[offset] > 0 ? inout[offset] : 0;
}

void ReLUDevice(Tensor *inout) {
  int N = inout->shape[0], C = inout->shape[1], H = inout->shape[2], W = inout->shape[3];
  int total_threads = N*C*H*W;
  int block_size = 256;
  int num_blocks = (total_threads + block_size - 1) / block_size;
  ReLUKernel<<<num_blocks, block_size>>>(inout->buf_gpu, N, C, H, W);
  cudaDeviceSynchronize();
}



__global__ void BatchNorm2dKernel(float *input, float *gamma, float *beta,
                                  float *running_mean, float *running_var, float *output,
                                  int N, int C, int H, int W,
                                  const float eps, const float momentum) {
  int n = blockIdx.x;
  int c = blockIdx.y * blockDim.x + threadIdx.x;

  if (c < C) {
    float mean = running_mean[c];
    float variance = running_var[c];

    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        for (int i = 0; i < N; ++i) {
          float x = input[i * C * H * W + c * H * W + h * W + w];
          float x_hat = (x - mean) / sqrt(variance + eps);
          output[i * C * H * W + c * H * W + h * W + w] =
              gamma[c] * x_hat + beta[c];
        }
      }
    }
  }
}

void BatchNorm2dDevice(Tensor *input, Tensor *gamma, Tensor *beta,
                       Tensor *running_mean, Tensor *running_var, Tensor *output,
                       const float eps, const float momentum
                       ) {

  int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  dim3 blockDim(256);
  dim3 gridDim(N, (C + blockDim.x - 1) / blockDim.x);

  BatchNorm2dKernel<<<gridDim, blockDim>>>(
    input->buf_gpu, gamma->buf_gpu, beta->buf_gpu, running_mean->buf_gpu, running_var->buf_gpu, output->buf_gpu,
    N, C, H, W, eps, momentum
  );
    cudaDeviceSynchronize();
}



__device__ float max4Device(float a, float b, float c, float d) {
  return fmaxf(fmaxf(a, b), fmaxf(c, d));
}
__global__ void MaxPool2dKernel(float *input, float *output, int C, int H, int W, int OH, int OW) {
  int oc = blockIdx.x;
  int oh = blockIdx.y;
  int ow = threadIdx.x;

  float in0 = input[oc * H * W + 2 * oh * W + 2 * ow];
  float in1 = input[oc * H * W + 2 * oh * W + 2 * ow + 1];
  float in2 = input[oc * H * W + (2 * oh + 1) * W + 2 * ow];
  float in3 = input[oc * H * W + (2 * oh + 1) * W + 2 * ow + 1];
  output[oc * OH * OW + oh * OW + ow] = max4Device(in0, in1, in2, in3);
}

void MaxPool2dDevice(Tensor *input, Tensor *output) {
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  dim3 blockDim(OW);
  dim3 gridDim(C, OH);
  MaxPool2dKernel<<<gridDim, blockDim>>>(input->buf_gpu, output->buf_gpu, C, H, W, OH, OW);
  cudaDeviceSynchronize();
}



__global__ void ConvTranspose2dKernel(
    float *input, float *weight, float *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S, int OH, int OW,
    int stride, int pad
) {
    const int oh = blockDim.x * blockIdx.x + threadIdx.x;
    const int ow = blockDim.y * blockIdx.y + threadIdx.y;
    const int k = blockIdx.z;
    if (oh >= OH || ow >= OW || k >= K) return;

    float sum = bias[k];

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    if ((oh + pad - r) % stride != 0) continue;
                    if ((ow + pad - s) % stride != 0) continue;
                    int h = (oh + pad - r) / stride;
                    int w = (ow + pad - s) / stride;
                    if (h < 0 || h >= H || w < 0 || w >= W) continue;
                    float input_val = input[(n * C + c) * H * W + h * W + w];
                    float weight_val = weight[(c * K + k) * R * S + r * S + s];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    output[(k * OH + oh) * OW + ow] = sum;
}

void ConvTranspose2dDevice(Tensor *input, Tensor *weight, Tensor *bias,
                           Tensor *output, int stride, int pad) {
    int N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
    int OH = output->shape[2], OW = output->shape[3];


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((OH + blockDim.x - 1) / blockDim.x, (OW + blockDim.y - 1) / blockDim.y, K);
    ConvTranspose2dKernel<<<gridDim, blockDim>>>(
        input->buf_gpu, weight->buf_gpu, bias->buf_gpu, output->buf_gpu,
        N, C, H, W, K, R, S, OH, OW, stride, pad
    );
    cudaDeviceSynchronize();
}


 __global__ void ConcatKernel(
    float *input1, float *input2, float *output,
    int C1, int H1, int W1, int C2, int H2, int W2,
    int OC, int OH, int OW
) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= OC * OH * OW) return;

  const int oc = tid / (OH * OW);
  const int oh = (tid / OW) % OH;
  const int ow = tid % OW;

  if (oc < OC / 2) {
    output[tid] = input2[oh * W2 + ow + oc * OH * OW];
  } else {
    const int ic = oc - OC / 2;
    if (ow < W1 && oh < H1) {
      output[tid] = input1[oh * W1 + ow + ic * H1 * W1];
    } else {
      output[tid] = 0.0;  // zero padding
    }
  }
}
void ConcatDevice(Tensor *input1, Tensor *input2, Tensor *output) {
  int C1 = input1->shape[1], H1 = input1->shape[2], W1 = input1->shape[3];
  int C2 = input2->shape[1], H2 = input2->shape[2], W2 = input2->shape[3];
  int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((OC * OH * OW + blockDim.x - 1) / blockDim.x);

  ConcatKernel<<<gridDim, blockDim>>>(
      input1->buf_gpu, input2->buf_gpu, output->buf_gpu,
      C1, H1, W1, C2, H2, W2, OC, OH, OW
  );
   cudaDeviceSynchronize();
}



// __global__ void ConcatKernel(
//     float *input1, float *input2, float *output,
//     int C1, int H1, int W1, int C2, int H2, int W2,
//     int OC, int OH, int OW) {
//   const int oc = blockDim.x * blockIdx.x + threadIdx.x;
//   if (oc >= OC) return;

//   if (oc < OC / 2) {
//     for (int oh = 0; oh < OH; ++oh) {
//       for (int ow = 0; ow < OW; ++ow) {
//         output[oc * OH * OW + oh * OW + ow] =
//             input2[oc * OH * OW + oh * OW + ow];
//       }
//     }
//   } else {
//     for (int oh = 0; oh < OH; ++oh) {
//       for (int ow = 0; ow < OW; ++ow) {
//         if (ow == OW - 1)
//           output[oc * OH * OW + oh * OW + ow] = 0.0;  // zero padding
//         else
//           output[oc * OH * OW + oh * OW + ow] =
//               input1[(oc - OC / 2) * H1 * W1 + oh * W1 + ow];
//       }
//     }
//   }
// }

// void ConcatDevice(Tensor *input1, Tensor *input2, Tensor *output) {
//   int C1 = input1->shape[1], H1 = input1->shape[2], W1 = input1->shape[3];
//   int C2 = input2->shape[1], H2 = input2->shape[2], W2 = input2->shape[3];
//   int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

//   dim3 blockDim(OC);
//   dim3 gridDim(1);

//   ConcatKernel<<<gridDim, blockDim>>>(
//       input1->buf_gpu, input2->buf_gpu, output->buf_gpu,
//       C1, H1, W1, C2, H2, W2, OC, OH, OW
//   );
//   cudaDeviceSynchronize();
// }

// forward declaration, prototype
void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias);
void ReLU(Tensor *inout);
void BatchNorm2d(Tensor *input, Tensor *gamma, Tensor *beta,
                 Tensor *running_mean, Tensor *running_var, Tensor *output,
                 const float eps, const float momentum);
void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *bias,
                     Tensor *output, int stride, int pad);
void MaxPool2d(Tensor *input, Tensor *output);
void Concat(Tensor *input1, Tensor *input2, Tensor *output);
void uNet_initialize(int, int, char *);
void uNet(Tensor *, Tensor *);
void uNet_finalize();

/*
 * uNet
 * This model identifies the boundaries of the cars in an image file (input.bin)
 * and removes the background.
 */






















void uNet(Tensor *inputN, Tensor *outputN, int N) {

  Tensor *input = new Tensor({BATCH_SIZE, 3, 128, 191});
  Tensor *output = new Tensor({BATCH_SIZE, 2, 128, 191});

  int step_size = (N+BATCH_SIZE-1)/BATCH_SIZE;
    for (int idx = 0; idx < N; ++idx) {

    cudaMemcpy(input->buf_gpu, 
     inputN->buf + (idx * 1 * 3 * 128 * 191),
     sizeof(float) * 1 * 3 * 128 * 191, 
      cudaMemcpyHostToDevice
    );

    // inc(n_channels, 64)
    Conv2dDevice(input, inc_double_conv_0_weight, NULL, inc_conv_0_output, 1, 1, 1,
           false);
    BatchNorm2dDevice(inc_conv_0_output, inc_double_conv_1_weight,
                inc_double_conv_1_bias, inc_batchnorm_0_running_mean,
                inc_batchnorm_0_running_var, inc_batchnorm_0_output, 1e-5, 0.1);
    ReLUDevice(inc_batchnorm_0_output);

    Conv2dDevice(inc_batchnorm_0_output, inc_double_conv_3_weight, NULL,
           inc_conv_1_output, 1, 1, 1, false);
    
    BatchNorm2dDevice(inc_conv_1_output, inc_double_conv_4_weight,
                inc_double_conv_4_bias, inc_batchnorm_1_running_mean,
                inc_batchnorm_1_running_var, inc_batchnorm_1_output, 1e-5, 0.1);
    
    ReLUDevice(inc_batchnorm_1_output);
    
    // down1(64, 128)
    MaxPool2dDevice(inc_batchnorm_1_output, down1_maxpool2d_0_output);

    Conv2dDevice(down1_maxpool2d_0_output, down1_maxpool_conv_1_double_conv_0_weight,
              NULL, down1_conv_0_output, 1, 1, 1, false);

    BatchNorm2dDevice(down1_conv_0_output, down1_maxpool_conv_1_double_conv_1_weight,
                down1_maxpool_conv_1_double_conv_1_bias,
                down1_batchnorm_0_running_mean, down1_batchnorm_0_running_var,
                down1_batchnorm_0_output, 1e-5, 0.1);

    ReLUDevice(down1_batchnorm_0_output);
    
    Conv2dDevice(down1_batchnorm_0_output, down1_maxpool_conv_1_double_conv_3_weight,
           NULL, down1_conv_1_output, 1, 1, 1, false);

    BatchNorm2dDevice(down1_conv_1_output, down1_maxpool_conv_1_double_conv_4_weight,
                down1_maxpool_conv_1_double_conv_4_bias,
                down1_batchnorm_1_running_mean, down1_batchnorm_1_running_var,
                down1_batchnorm_1_output, 1e-5, 0.1);

    ReLUDevice(down1_batchnorm_1_output);

    // down2(128, 256)
    MaxPool2dDevice(down1_batchnorm_1_output, down2_maxpool2d_0_output);

    Conv2dDevice(down2_maxpool2d_0_output, down2_maxpool_conv_1_double_conv_0_weight,
           NULL, down2_conv_0_output, 1, 1, 1, false);

    BatchNorm2dDevice(down2_conv_0_output, down2_maxpool_conv_1_double_conv_1_weight,
                down2_maxpool_conv_1_double_conv_1_bias,
                down2_batchnorm_0_running_mean, down2_batchnorm_0_running_var,
                down2_batchnorm_0_output, 1e-5, 0.1);
    
    ReLUDevice(down2_batchnorm_0_output);
    
    Conv2dDevice(down2_batchnorm_0_output, down2_maxpool_conv_1_double_conv_3_weight,
           NULL, down2_conv_1_output, 1, 1, 1, false);
    
    BatchNorm2dDevice(down2_conv_1_output, down2_maxpool_conv_1_double_conv_4_weight,
                down2_maxpool_conv_1_double_conv_4_bias,
                down2_batchnorm_1_running_mean, down2_batchnorm_1_running_var,
                down2_batchnorm_1_output, 1e-5, 0.1);
    
    ReLUDevice(down2_batchnorm_1_output);
    
    // up1(256, 128), (up2_concat_0_output, down1_batchnorm_1_output)
    ConvTranspose2dDevice(down2_batchnorm_1_output, up1_up_weight, up1_up_bias,
                    up1_convt_0_output, 2, 0);
    
    ConcatDevice(up1_convt_0_output, down1_batchnorm_1_output, up1_concat_0_output);

    Conv2dDevice(up1_concat_0_output, up1_conv_double_conv_0_weight, NULL,
           up1_conv_0_output, 1, 1, 1, false);

    BatchNorm2dDevice(up1_conv_0_output, up1_conv_double_conv_1_weight,
                up1_conv_double_conv_1_bias, up1_batchnorm_0_running_mean,
                up1_batchnorm_0_running_var, up1_batchnorm_0_output, 1e-5, 0.1);

    ReLUDevice(up1_batchnorm_0_output);

    Conv2dDevice(up1_batchnorm_0_output, up1_conv_double_conv_3_weight, NULL,
           up1_conv_1_output, 1, 1, 1, false);

    BatchNorm2dDevice(up1_conv_1_output, up1_conv_double_conv_4_weight,
                up1_conv_double_conv_4_bias, up1_batchnorm_1_running_mean,
                up1_batchnorm_1_running_var, up1_batchnorm_1_output, 1e-5, 0.1);
    
    ReLUDevice(up1_batchnorm_1_output);


    // up2(128, 64), (up1_concat_0_output, inc_batchnorm_1_output)
    ConvTranspose2dDevice(up1_batchnorm_1_output, up2_up_weight, up2_up_bias,
                    up2_convt_0_output, 2, 0);

    ConcatDevice(up2_convt_0_output, inc_batchnorm_1_output, up2_concat_0_output);
    Conv2dDevice(up2_concat_0_output, up2_conv_double_conv_0_weight, NULL,
           up2_conv_0_output, 1, 1, 1, false);
    BatchNorm2dDevice(up2_conv_0_output, up2_conv_double_conv_1_weight,
                up2_conv_double_conv_1_bias, up2_batchnorm_0_running_mean,
                up2_batchnorm_0_running_var, up2_batchnorm_0_output, 1e-5, 0.1);
    ReLUDevice(up2_batchnorm_0_output);
    Conv2dDevice(up2_batchnorm_0_output, up2_conv_double_conv_3_weight, NULL,
           up2_conv_1_output, 1, 1, 1, false);
    BatchNorm2dDevice(up2_conv_1_output, up2_conv_double_conv_4_weight,
                up2_conv_double_conv_4_bias, up2_batchnorm_1_running_mean,
                up2_batchnorm_1_running_var, up2_batchnorm_1_output, 1e-5, 0.1);
    ReLUDevice(up2_batchnorm_1_output);

    // outc(64, 2)
    Conv2dDevice(up2_batchnorm_1_output, outc_conv_weight, outc_conv_bias, output, 1,
           0, 1, true);

    cudaMemcpy(
      outputN->buf + (idx * 1 * 2 * 128 * 191),
      output->buf_gpu, 
      sizeof(float) * 1 * 2 * 128 * 191, 
      cudaMemcpyDeviceToHost
    );

    }

}






/* Operations */

/*
 * Convolution
 * input shape = (N, C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {

  
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[0], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
              "[Conv2d] Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1,
              "[Conv2d] Output width mismatch");
  CHECK_ERROR(weight->shape[1] == C && (!has_bias || bias->shape[0] == K) &&
                  output->shape[1] == K,
              "[Conv2d] Channel size mismatch");

#ifdef TEST
#pragma omp parallel for
#endif
  for (int k = 0; k < K; ++k) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float o = has_bias ? bias->buf[k] : 0;
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              int h = oh * stride - pad + r * dilation;
              int w = ow * stride - pad + s * dilation;
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input->buf[c * H * W + h * W + w];
              float f = weight->buf[k * C * R * S + c * R * S + r * S + s];
              o += i * f;
            }
          }
        }
        output->buf[k * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}

/*
 * ReLU
 * input shape = (N, C, H, W)
 * output shape = (N, C, H, W)
 * Formula: y = max(x, 0)
 */
void ReLU(Tensor *inout) {
  int C = inout->shape[1], H = inout->shape[2], W = inout->shape[3];

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int idx = c * H * W + h * W + w;
        inout->buf[idx] = inout->buf[idx] > 0 ? inout->buf[idx] : 0;
      }
    }
  }
}

/*
 * Batch Normaliztion
 * input shape = (N, C, H, W)
 * gamma shape = (C)
 * beta shape = (C)
 * output shape = (N, C, H, W)
 */
void BatchNorm2d(Tensor *input, Tensor *gamma, Tensor *beta,
                 Tensor *running_mean, Tensor *running_var, Tensor *output,
                 const float eps, const float momentum) {
  int N = input->shape[0], C = input->shape[1], H = input->shape[2],
      W = input->shape[3];

  CHECK_ERROR(gamma->shape[0] == C && beta->shape[0] == C,
              "[BatchNorm2d] gamma, beta shape mismatch");
  CHECK_ERROR(
      output->shape[1] == C && output->shape[2] == H && output->shape[3] == W,
      "[BatchNorm2d] Output shape mismatch");

  for (int c = 0; c < C; ++c) {
    for (int n = 0; n < N; ++n) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          float mean = running_mean->buf[c];
          float variance = running_var->buf[c];
          float x = input->buf[n * C * H * W + c * H * W + h * W + w];
          float x_hat = (x - mean) / sqrt(variance + eps);
          output->buf[n * C * H * W + c * H * W + h * W + w] =
              gamma->buf[c] * x_hat + beta->buf[c];
        }
      }
    }
  }
}

/*
 * Transposed convolution
 * input shape = (N, C, H, W)
 * weight shape = (C, K, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */
void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *bias,
                     Tensor *output, int stride, int pad) {
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R,
              "[ConvT2d] Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S,
              "[ConvT2d] Output width mismatch");
  CHECK_ERROR(
      weight->shape[0] == C && bias->shape[0] == K && output->shape[1] == K,
      "[ConvT2d] Channel size mismatch");

  for (int k = 0; k < K; ++k) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float o = bias->buf[k];
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              if ((oh + pad - r) % stride != 0) continue;
              if ((ow + pad - s) % stride != 0) continue;
              int h = (oh + pad - r) / stride;
              int w = (ow + pad - s) / stride;
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input->buf[c * H * W + h * W + w];
              float f = weight->buf[c * K * R * S + k * R * S + r * S + s];
              o += i * f;
            }
          }
        }
        output->buf[k * OH * OW + oh * OW + ow] = o;
      }
    }
  }
}

float max4(float in0, float in1, float in2, float in3) {
  float max = in0;

  if (in1 > max) max = in1;
  if (in2 > max) max = in2;
  if (in3 > max) max = in3;
  return max;
}

/*
 * MaxPool2d
 * input shape = (N, C, H, W)
 * output shape = (N, OC, OH, OW)
 *   where OH = H / 2
 *         OW = W / 2
 */
void MaxPool2d(Tensor *input, Tensor *output) {
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OW == W / 2, "[MaxPool2d] Output width mismatch");
  CHECK_ERROR(OH == H / 2, "[MaxPool2d] Output height mismatch");
  CHECK_ERROR(OC == C, "[MaxPool2d] Output channel mismatch");

  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float in0 = input->buf[oc * H * W + 2 * oh * W + 2 * ow];
        float in1 = input->buf[oc * H * W + 2 * oh * W + 2 * ow + 1];
        float in2 = input->buf[oc * H * W + (2 * oh + 1) * W + 2 * ow];
        float in3 = input->buf[oc * H * W + (2 * oh + 1) * W + 2 * ow + 1];
        output->buf[oc * OH * OW + oh * OW + ow] = max4(in0, in1, in2, in3);
      }
    }
  }
}

/*
 * Concat
 * input1 shape = (N, C1, H1, W1)
 * input2 shape = (N, C2, H2, W2)
 * output shape = (N, OC, OH, OW)
 *   where OH = H2, H1
 *         OW = W2 = W1 + 1
 */
void Concat(Tensor *input1, Tensor *input2, Tensor *output) {
  int C1 = input1->shape[1], H1 = input1->shape[2], W1 = input1->shape[3];
  int C2 = input2->shape[1], H2 = input2->shape[2], W2 = input2->shape[3];
  int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OC == C1 * 2 && OC == C2 * 2, "[Concat] Output channel mismatch");
  CHECK_ERROR(OW == W1 + 1 && OW == W2, "[Concat] Output width mismatch");
  CHECK_ERROR(OH == H1 && OH == H2, "[Concat] Output height mismatch");

  for (int oc = 0; oc < OC / 2; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        output->buf[oc * OH * OW + oh * OW + ow] =
            input2->buf[oc * OH * OW + oh * OW + ow];
      }
    }
  }

  for (int oc = OC / 2; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        if (ow == OW - 1)
          output->buf[oc * OH * OW + oh * OW + ow] = 0.0;  // zero padding
        else
          output->buf[oc * OH * OW + oh * OW + ow] =
              input1->buf[(oc - OC / 2) * H1 * W1 + oh * W1 + ow];
      }
    }
  }
}

/*
 * uNet_initialize
 * Initialize the model. Do input-independent job here.
 */
void uNet_initialize(int N, char *parameter_fname) {
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *) read_binary(parameter_fname, &parameter_binary_size);

  // Parameters
  inc_double_conv_0_weight = new Tensor({64, 3, 3, 3}, parameter + OFFSET0);
  inc_double_conv_1_weight = new Tensor({64}, parameter + OFFSET1);
  inc_double_conv_1_bias = new Tensor({64}, parameter + OFFSET2);
  inc_double_conv_3_weight = new Tensor({64, 64, 3, 3}, parameter + OFFSET3);
  inc_double_conv_4_weight = new Tensor({64}, parameter + OFFSET4);
  inc_double_conv_4_bias = new Tensor({64}, parameter + OFFSET5);
  down1_maxpool_conv_1_double_conv_0_weight =
      new Tensor({128, 64, 3, 3}, parameter + OFFSET6);
  down1_maxpool_conv_1_double_conv_1_weight =
      new Tensor({128}, parameter + OFFSET7);
  down1_maxpool_conv_1_double_conv_1_bias =
      new Tensor({128}, parameter + OFFSET8);
  down1_maxpool_conv_1_double_conv_3_weight =
      new Tensor({128, 128, 3, 3}, parameter + OFFSET9);
  down1_maxpool_conv_1_double_conv_4_weight =
      new Tensor({128}, parameter + OFFSET10);
  down1_maxpool_conv_1_double_conv_4_bias =
      new Tensor({128}, parameter + OFFSET11);
  down2_maxpool_conv_1_double_conv_0_weight =
      new Tensor({256, 128, 3, 3}, parameter + OFFSET12);
  down2_maxpool_conv_1_double_conv_1_weight =
      new Tensor({256}, parameter + OFFSET13);
  down2_maxpool_conv_1_double_conv_1_bias =
      new Tensor({256}, parameter + OFFSET14);
  down2_maxpool_conv_1_double_conv_3_weight =
      new Tensor({256, 256, 3, 3}, parameter + OFFSET15);
  down2_maxpool_conv_1_double_conv_4_weight =
      new Tensor({256}, parameter + OFFSET16);
  down2_maxpool_conv_1_double_conv_4_bias =
      new Tensor({256}, parameter + OFFSET17);
  up1_up_weight = new Tensor({256, 128, 2, 2}, parameter + OFFSET18);
  up1_up_bias = new Tensor({128}, parameter + OFFSET19);
  up1_conv_double_conv_0_weight =
      new Tensor({128, 256, 3, 3}, parameter + OFFSET20);
  up1_conv_double_conv_1_weight = new Tensor({128}, parameter + OFFSET21);
  up1_conv_double_conv_1_bias = new Tensor({128}, parameter + OFFSET22);
  up1_conv_double_conv_3_weight =
      new Tensor({128, 128, 3, 3}, parameter + OFFSET23);
  up1_conv_double_conv_4_weight = new Tensor({128}, parameter + OFFSET24);
  up1_conv_double_conv_4_bias = new Tensor({128}, parameter + OFFSET25);
  up2_up_weight = new Tensor({128, 64, 2, 2}, parameter + OFFSET26);
  up2_up_bias = new Tensor({64}, parameter + OFFSET27);
  up2_conv_double_conv_0_weight =
      new Tensor({64, 128, 3, 3}, parameter + OFFSET28);
  up2_conv_double_conv_1_weight = new Tensor({64}, parameter + OFFSET29);
  up2_conv_double_conv_1_bias = new Tensor({64}, parameter + OFFSET30);
  up2_conv_double_conv_3_weight =
      new Tensor({64, 64, 3, 3}, parameter + OFFSET31);
  up2_conv_double_conv_4_weight = new Tensor({64}, parameter + OFFSET32);
  up2_conv_double_conv_4_bias = new Tensor({64}, parameter + OFFSET33);
  outc_conv_weight = new Tensor({2, 64, 1, 1}, parameter + OFFSET34);
  outc_conv_bias = new Tensor({2}, parameter + OFFSET35);
  inc_batchnorm_0_running_mean = new Tensor({64}, parameter + OFFSET36);
  inc_batchnorm_0_running_var = new Tensor({64}, parameter + OFFSET37);
  inc_batchnorm_1_running_mean = new Tensor({64}, parameter + OFFSET38);
  inc_batchnorm_1_running_var = new Tensor({64}, parameter + OFFSET39);
  down1_batchnorm_0_running_mean = new Tensor({128}, parameter + OFFSET40);
  down1_batchnorm_0_running_var = new Tensor({128}, parameter + OFFSET41);
  down1_batchnorm_1_running_mean = new Tensor({128}, parameter + OFFSET42);
  down1_batchnorm_1_running_var = new Tensor({128}, parameter + OFFSET43);
  down2_batchnorm_0_running_mean = new Tensor({256}, parameter + OFFSET44);
  down2_batchnorm_0_running_var = new Tensor({256}, parameter + OFFSET45);
  down2_batchnorm_1_running_mean = new Tensor({256}, parameter + OFFSET46);
  down2_batchnorm_1_running_var = new Tensor({256}, parameter + OFFSET47);
  up1_batchnorm_0_running_mean = new Tensor({128}, parameter + OFFSET48);
  up1_batchnorm_0_running_var = new Tensor({128}, parameter + OFFSET49);
  up1_batchnorm_1_running_mean = new Tensor({128}, parameter + OFFSET50);
  up1_batchnorm_1_running_var = new Tensor({128}, parameter + OFFSET51);
  up2_batchnorm_0_running_mean = new Tensor({64}, parameter + OFFSET52);
  up2_batchnorm_0_running_var = new Tensor({64}, parameter + OFFSET53);
  up2_batchnorm_1_running_mean = new Tensor({64}, parameter + OFFSET54);
  up2_batchnorm_1_running_var = new Tensor({64}, parameter + OFFSET55);

  // Activations
  inc_conv_0_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  inc_batchnorm_0_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  inc_conv_1_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  inc_batchnorm_1_output = new Tensor({BATCH_SIZE, 64, 128, 191});

  down1_maxpool2d_0_output = new Tensor({BATCH_SIZE, 64, 64, 95});
  down1_conv_0_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  down1_batchnorm_0_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  down1_conv_1_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  down1_batchnorm_1_output = new Tensor({BATCH_SIZE, 128, 64, 95});

  down2_maxpool2d_0_output = new Tensor({BATCH_SIZE, 128, 32, 47});
  down2_conv_0_output = new Tensor({BATCH_SIZE, 256, 32, 47});
  down2_batchnorm_0_output = new Tensor({BATCH_SIZE, 256, 32, 47});
  down2_conv_1_output = new Tensor({BATCH_SIZE, 256, 32, 47});
  down2_batchnorm_1_output = new Tensor({BATCH_SIZE, 256, 32, 47});

  up1_convt_0_output = new Tensor({BATCH_SIZE, 128, 64, 94});
  up1_concat_0_output = new Tensor({BATCH_SIZE, 256, 64, 95});
  up1_conv_0_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  up1_batchnorm_0_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  up1_conv_1_output = new Tensor({BATCH_SIZE, 128, 64, 95});
  up1_batchnorm_1_output = new Tensor({BATCH_SIZE, 128, 64, 95});

  up2_convt_0_output = new Tensor({BATCH_SIZE, 64, 128, 190});
  up2_concat_0_output = new Tensor({BATCH_SIZE, 128, 128, 191});
  up2_conv_0_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  up2_batchnorm_0_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  up2_conv_1_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  up2_batchnorm_1_output = new Tensor({BATCH_SIZE, 64, 128, 191});
  outc_conv_0_output = new Tensor({BATCH_SIZE, 2, 128, 191});
}

/*
 * uNet_finalize
 * Finalize the model.
 */
void uNet_finalize() {
  // delete parameters
  delete inc_double_conv_0_weight;
  delete inc_double_conv_1_weight;
  delete inc_double_conv_1_bias;
  delete inc_double_conv_3_weight;
  delete inc_double_conv_4_weight;
  delete inc_double_conv_4_bias;
  delete down1_maxpool_conv_1_double_conv_0_weight;
  delete down1_maxpool_conv_1_double_conv_1_weight;
  delete down1_maxpool_conv_1_double_conv_1_bias;
  delete down1_maxpool_conv_1_double_conv_3_weight;
  delete down1_maxpool_conv_1_double_conv_4_weight;
  delete down1_maxpool_conv_1_double_conv_4_bias;
  delete down2_maxpool_conv_1_double_conv_0_weight;
  delete down2_maxpool_conv_1_double_conv_1_weight;
  delete down2_maxpool_conv_1_double_conv_1_bias;
  delete down2_maxpool_conv_1_double_conv_3_weight;
  delete down2_maxpool_conv_1_double_conv_4_weight;
  delete down2_maxpool_conv_1_double_conv_4_bias;
  delete up1_up_weight;
  delete up1_up_bias;
  delete up1_conv_double_conv_0_weight;
  delete up1_conv_double_conv_1_weight;
  delete up1_conv_double_conv_1_bias;
  delete up1_conv_double_conv_3_weight;
  delete up1_conv_double_conv_4_weight;
  delete up1_conv_double_conv_4_bias;
  delete up2_up_weight;
  delete up2_up_bias;
  delete up2_conv_double_conv_0_weight;
  delete up2_conv_double_conv_1_weight;
  delete up2_conv_double_conv_1_bias;
  delete up2_conv_double_conv_3_weight;
  delete up2_conv_double_conv_4_weight;
  delete up2_conv_double_conv_4_bias;
  delete outc_conv_weight;
  delete outc_conv_bias;
  delete inc_batchnorm_0_running_mean;
  delete inc_batchnorm_0_running_var;
  delete down1_batchnorm_0_running_mean;
  delete down1_batchnorm_0_running_var;
  delete down2_batchnorm_0_running_mean;
  delete down2_batchnorm_0_running_var;
  delete up1_batchnorm_0_running_mean;
  delete up1_batchnorm_0_running_var;
  delete up2_batchnorm_0_running_mean;
  delete up2_batchnorm_0_running_var;
  delete inc_batchnorm_1_running_mean;
  delete inc_batchnorm_1_running_var;
  delete down1_batchnorm_1_running_mean;
  delete down1_batchnorm_1_running_var;
  delete down2_batchnorm_1_running_mean;
  delete down2_batchnorm_1_running_var;
  delete up1_batchnorm_1_running_mean;
  delete up1_batchnorm_1_running_var;
  delete up2_batchnorm_1_running_mean;
  delete up2_batchnorm_1_running_var;

  // delete activations
  delete inc_conv_0_output;
  delete inc_batchnorm_0_output;
  delete inc_conv_1_output;
  delete inc_batchnorm_1_output;
  delete down1_maxpool2d_0_output;
  delete down1_conv_0_output;
  delete down1_batchnorm_0_output;
  delete down1_conv_1_output;
  delete down1_batchnorm_1_output;
  delete down2_maxpool2d_0_output;
  delete down2_conv_0_output;
  delete down2_batchnorm_0_output;
  delete down2_conv_1_output;
  delete down2_batchnorm_1_output;
  delete up1_convt_0_output;
  delete up1_concat_0_output;
  delete up1_conv_0_output;
  delete up1_batchnorm_0_output;
  delete up1_conv_1_output;
  delete up1_batchnorm_1_output;
  delete up2_convt_0_output;
  delete up2_concat_0_output;
  delete up2_conv_0_output;
  delete up2_batchnorm_0_output;
  delete up2_conv_1_output;
  delete up2_batchnorm_1_output;
  delete outc_conv_0_output;
}








/*

  MULTI THREADING 
      blockDim : thread의 인덱스를 결정하는 놈
      gridDim : block의 인덱스를 결정하는 놈 
      
      1. 인덱스 공간의 전체 공간 생각 (H*W)
      2. 그 후 블록 크기 생각 (5*4)
      3. 그러면 블록 개수는 (3*3)이 될 것
        - 블록 개수 -> grid dimension. 블록이 몇개 있느냐
        - gridDim(3,3)

      threadIdx : 블록 안에서의 상대적인 위치
      blockIdx : 블록 자체의 전체 그리드에서의 상대적 위치.


  SYNCTHREADING
      유의할 점
        - 일부 스레드만 __syncthread() 하면 안 됨
        - kernel에서 윗 부분에서 인덱스 계산해서 return 해버리는 경우! (자주 발생하는 실수)
          - 아래에 있는 syncthread 함수가 진행되지 못하면서 stall 되어버리는 문제

  ATOMIC OPERATION
      - 누적합 등에 사용

      int, float, double 등  
        atomicAdd(int* address, int val);   -> address에 val 더해라
        atomicSub(int* address, int val);   -> 뺄셈
        atomicExch(int* address, int val);  -> 교환

        atomicMin
        atomicMax
        atomicInc : increase, 0이 아니면 하나씩 증가
        atomicDec : decrease

      - 이걸로 block 간 동기화 할 생각 하지 말라! (하지말라면 하지마..)
        - block 내 동기화만 수행할 것

  TERMS
      - shared => local memory라서 L로 표현함



  STRATEGY
      1. CUDA thread 하나가 out_channels의 하나를 처리하도록 만드는 방법
        - thread 하나가 element OH*OW 개 처리
        - thread 개수 : N*K (out_channels)
      2. CUDA thread 하나가 출력 데이터의 원소 하나를 처리하도록 만드는 방법
        - thread 개수 : N*K*OH*OW
        - 인덱스 공간 1차원으로 만들고 커널 내부에서 4차원으로 변환
          ex. gridDim(ceil(N*K*OH*OW/ 512), 1, 1)  
              blockDim(512)



  NOTE 
    - output 내보내는 conv만 padding이 없고 bias가 존재함
    - 나머지는 padding없고 bias 없음 !

    
  STREAM
      - kernel<<<gridDim, blockDim, stream idx, stream>>> 
      - 비동기적으로 실행
      - cudaStreamSynchronize(stream)












  kernel api는 CUDA_CHECK 사용 불가 !
  cudaGetLastError() 로 에러처리

  4행 5열 -> (5, 4)  !  4, 5 아님
  threadIdx.x : 5
  threadIdx.y : 4

  gridDim(3, 3), blockDim(5,4) -> 4행 5열!!!!
  이미지 8개 -> 8개 블록에 올려버리기?

  v100 80개 SM~ thread block은 더 많아야 될 것

Number of devices: 4
        device 0:
                name: Tesla V100-PCIE-32GB
                multiProcessorCount: 80
                maxThreadspPerBlock: 1024
                totalGlobalMem: 34089664512
                sharedMemPerBlock: 49152
        device 1:
                name: Tesla V100-PCIE-32GB
                multiProcessorCount: 80
                maxThreadspPerBlock: 1024
                totalGlobalMem: 34089664512
                sharedMemPerBlock: 49152
        device 2:
                name: Tesla V100-PCIE-32GB
                multiProcessorCount: 80
                maxThreadspPerBlock: 1024
                totalGlobalMem: 34089664512
                sharedMemPerBlock: 49152
        device 3:
                name: Tesla V100-PCIE-32GB
                multiProcessorCount: 80
                maxThreadspPerBlock: 1024
                totalGlobalMem: 34089664512
                sharedMemPerBlock: 49152



  비동기 API -> cudaMemcpyAsync -> syhcronize 한거랑 같은 효과


*/








      // Conv2dKernel<<<grid_gen(input, BLOCK_SIZE), blockDim>>>(input, inc_double_conv_0_weight, NULL, inc_conv_0_output, 1, 1, 1,
      //       false);
      // BatchNorm2dKernel<<<grid_gen(inc_conv_0_output, BLOCK_SIZE), blockDim>>>(inc_conv_0_output, inc_double_conv_1_weight,
      //             inc_double_conv_1_bias, inc_batchnorm_0_running_mean,
      //             inc_batchnorm_0_running_var, inc_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(inc_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(inc_batchnorm_0_output);
      // Conv2dKernel<<<grid_gen(inc_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(inc_batchnorm_0_output, inc_double_conv_3_weight, NULL,
      //       inc_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(inc_conv_1_output, BLOCK_SIZE), blockDim>>>(inc_conv_1_output, inc_double_conv_4_weight,
      //             inc_double_conv_4_bias, inc_batchnorm_1_running_mean,
      //             inc_batchnorm_1_running_var, inc_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(inc_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(inc_batchnorm_1_output);

      // // down1Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(64, 128)
      // MaxPool2dKernel<<<grid_gen(inc_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(inc_batchnorm_1_output, down1_maxpool2d_0_output);
      // Conv2dKernel<<<grid_gen(down1_maxpool2d_0_output, BLOCK_SIZE), blockDim>>>(down1_maxpool2d_0_output, down1_maxpool_conv_1_double_conv_0_weight,
      //       NULL, down1_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(down1_conv_0_output, BLOCK_SIZE), blockDim>>>(down1_conv_0_output, down1_maxpool_conv_1_double_conv_1_weight,
      //             down1_maxpool_conv_1_double_conv_1_bias,
      //             down1_batchnorm_0_running_mean, down1_batchnorm_0_running_var,
      //             down1_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(down1_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(down1_batchnorm_0_output);
      // Conv2dKernel<<<grid_gen(down1_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(down1_batchnorm_0_output, down1_maxpool_conv_1_double_conv_3_weight,
      //       NULL, down1_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(down1_conv_1_output, BLOCK_SIZE), blockDim>>>(down1_conv_1_output, down1_maxpool_conv_1_double_conv_4_weight,
      //             down1_maxpool_conv_1_double_conv_4_bias,
      //             down1_batchnorm_1_running_mean, down1_batchnorm_1_running_var,
      //             down1_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(down1_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(down1_batchnorm_1_output);

      // // down2Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(128, 256)
      // MaxPool2dKernel<<<grid_gen(down1_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(down1_batchnorm_1_output, down2_maxpool2d_0_output);
      // Conv2dKernel<<<grid_gen(down2_maxpool2d_0_output, BLOCK_SIZE), blockDim>>>(down2_maxpool2d_0_output, down2_maxpool_conv_1_double_conv_0_weight,
      //       NULL, down2_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(down2_conv_0_output, BLOCK_SIZE), blockDim>>>(down2_conv_0_output, down2_maxpool_conv_1_double_conv_1_weight,
      //             down2_maxpool_conv_1_double_conv_1_bias,
      //             down2_batchnorm_0_running_mean, down2_batchnorm_0_running_var,
      //             down2_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(down2_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(down2_batchnorm_0_output);
      // Conv2dKernel<<<grid_gen(down2_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(down2_batchnorm_0_output, down2_maxpool_conv_1_double_conv_3_weight,
      //       NULL, down2_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(down2_conv_1_output, BLOCK_SIZE), blockDim>>>(down2_conv_1_output, down2_maxpool_conv_1_double_conv_4_weight,
      //             down2_maxpool_conv_1_double_conv_4_bias,
      //             down2_batchnorm_1_running_mean, down2_batchnorm_1_running_var,
      //             down2_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(down2_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(down2_batchnorm_1_output);

      // // up1Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(256, 128), Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(up2_concat_0_output, down1_batchnorm_1_output)
      // ConvTranspose2dKernel<<<grid_gen(down2_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(down2_batchnorm_1_output, up1_up_weight, up1_up_bias,
      //                 up1_convt_0_output, 2, 0);
      // ConcatKernel<<<grid_gen(up1_convt_0_output, BLOCK_SIZE), blockDim>>>(up1_convt_0_output, down1_batchnorm_1_output, up1_concat_0_output);
      // Conv2dKernel<<<grid_gen(up1_concat_0_output, BLOCK_SIZE), blockDim>>>(up1_concat_0_output, up1_conv_double_conv_0_weight, NULL,
      //       up1_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(up1_conv_0_output, BLOCK_SIZE), blockDim>>>(up1_conv_0_output, up1_conv_double_conv_1_weight,
      //             up1_conv_double_conv_1_bias, up1_batchnorm_0_running_mean,
      //             up1_batchnorm_0_running_var, up1_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(up1_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(up1_batchnorm_0_output);
      // Conv2dKernel<<<grid_gen(up1_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(up1_batchnorm_0_output, up1_conv_double_conv_3_weight, NULL,
      //       up1_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(up1_conv_1_output, BLOCK_SIZE), blockDim>>>(up1_conv_1_output, up1_conv_double_conv_4_weight,
      //             up1_conv_double_conv_4_bias, up1_batchnorm_1_running_mean,
      //             up1_batchnorm_1_running_var, up1_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(up1_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(up1_batchnorm_1_output);

      // // up2Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(128, 64), Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(up1_concat_0_output, inc_batchnorm_1_output)
      // ConvTranspose2dKernel<<<grid_gen(up1_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(up1_batchnorm_1_output, up2_up_weight, up2_up_bias,
      //                 up2_convt_0_output, 2, 0);
      // ConcatKernel<<<grid_gen(up2_convt_0_output, BLOCK_SIZE), blockDim>>>(up2_convt_0_output, inc_batchnorm_1_output, up2_concat_0_output);
      // Conv2dKernel<<<grid_gen(up2_concat_0_output, BLOCK_SIZE), blockDim>>>(up2_concat_0_output, up2_conv_double_conv_0_weight, NULL,
      //       up2_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(up2_conv_0_output, BLOCK_SIZE), blockDim>>>(up2_conv_0_output, up2_conv_double_conv_1_weight,
      //             up2_conv_double_conv_1_bias, up2_batchnorm_0_running_mean,
      //             up2_batchnorm_0_running_var, up2_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(up2_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(up2_batchnorm_0_output);
      // Conv2dKernel<<<grid_gen(up2_batchnorm_0_output, BLOCK_SIZE), blockDim>>>(up2_batchnorm_0_output, up2_conv_double_conv_3_weight, NULL,
      //       up2_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<grid_gen(up2_conv_1_output, BLOCK_SIZE), blockDim>>>(up2_conv_1_output, up2_conv_double_conv_4_weight,
      //             up2_conv_double_conv_4_bias, up2_batchnorm_1_running_mean,
      //             up2_batchnorm_1_running_var, up2_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<grid_gen(up2_batchnorm_1_output, BLOCK_SIZE), blockDim>>>(up2_batchnorm_1_output);
      // cudaStreamSynchronize(0);

      // // outcKernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(64, 2)
      // Conv2dKernel<<<1, 1>>>(up2_batchnorm_1_output, outc_conv_weight, outc_conv_bias, output, 1,
      //       0, 1, true);















// <<<1, 1>>> 모음
      // // incKernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(n_channels, 64)
      // Conv2dKernel<<<1, 1>>>(input, inc_double_conv_0_weight, NULL, inc_conv_0_output, 1, 1, 1,
      //       false);
      // BatchNorm2dKernel<<<1, 1 >>>(inc_conv_0_output, inc_double_conv_1_weight,
      //             inc_double_conv_1_bias, inc_batchnorm_0_running_mean,
      //             inc_batchnorm_0_running_var, inc_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(inc_batchnorm_0_output);
      // Conv2dKernel<<<1, 1>>>(inc_batchnorm_0_output, inc_double_conv_3_weight, NULL,
      //       inc_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(inc_conv_1_output, inc_double_conv_4_weight,
      //             inc_double_conv_4_bias, inc_batchnorm_1_running_mean,
      //             inc_batchnorm_1_running_var, inc_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(inc_batchnorm_1_output);

      // // down1Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(64, 128)
      // MaxPool2dKernel<<<1, 1>>>(inc_batchnorm_1_output, down1_maxpool2d_0_output);
      // Conv2dKernel<<<1, 1>>>(down1_maxpool2d_0_output, down1_maxpool_conv_1_double_conv_0_weight,
      //       NULL, down1_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(down1_conv_0_output, down1_maxpool_conv_1_double_conv_1_weight,
      //             down1_maxpool_conv_1_double_conv_1_bias,
      //             down1_batchnorm_0_running_mean, down1_batchnorm_0_running_var,
      //             down1_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(down1_batchnorm_0_output);
      // Conv2dKernel<<<1, 1>>>(down1_batchnorm_0_output, down1_maxpool_conv_1_double_conv_3_weight,
      //       NULL, down1_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(down1_conv_1_output, down1_maxpool_conv_1_double_conv_4_weight,
      //             down1_maxpool_conv_1_double_conv_4_bias,
      //             down1_batchnorm_1_running_mean, down1_batchnorm_1_running_var,
      //             down1_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(down1_batchnorm_1_output);

      // // down2Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(128, 256)
      // MaxPool2dKernel<<<1, 1>>>(down1_batchnorm_1_output, down2_maxpool2d_0_output);
      // Conv2dKernel<<<1, 1>>>(down2_maxpool2d_0_output, down2_maxpool_conv_1_double_conv_0_weight,
      //       NULL, down2_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(down2_conv_0_output, down2_maxpool_conv_1_double_conv_1_weight,
      //             down2_maxpool_conv_1_double_conv_1_bias,
      //             down2_batchnorm_0_running_mean, down2_batchnorm_0_running_var,
      //             down2_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(down2_batchnorm_0_output);
      // Conv2dKernel<<<1, 1>>>(down2_batchnorm_0_output, down2_maxpool_conv_1_double_conv_3_weight,
      //       NULL, down2_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(down2_conv_1_output, down2_maxpool_conv_1_double_conv_4_weight,
      //             down2_maxpool_conv_1_double_conv_4_bias,
      //             down2_batchnorm_1_running_mean, down2_batchnorm_1_running_var,
      //             down2_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(down2_batchnorm_1_output);

      // // up1Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(256, 128), Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(up2_concat_0_output, down1_batchnorm_1_output)
      // ConvTranspose2dKernel<<<1, 1>>>(down2_batchnorm_1_output, up1_up_weight, up1_up_bias,
      //                 up1_convt_0_output, 2, 0);
      // ConcatKernel<<<1, 1>>>(up1_convt_0_output, down1_batchnorm_1_output, up1_concat_0_output);
      // Conv2dKernel<<<1, 1>>>(up1_concat_0_output, up1_conv_double_conv_0_weight, NULL,
      //       up1_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(up1_conv_0_output, up1_conv_double_conv_1_weight,
      //             up1_conv_double_conv_1_bias, up1_batchnorm_0_running_mean,
      //             up1_batchnorm_0_running_var, up1_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(up1_batchnorm_0_output);
      // Conv2dKernel<<<1, 1>>>(up1_batchnorm_0_output, up1_conv_double_conv_3_weight, NULL,
      //       up1_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(up1_conv_1_output, up1_conv_double_conv_4_weight,
      //             up1_conv_double_conv_4_bias, up1_batchnorm_1_running_mean,
      //             up1_batchnorm_1_running_var, up1_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(up1_batchnorm_1_output);

      // // up2Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(128, 64), Kernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(up1_concat_0_output, inc_batchnorm_1_output)
      // ConvTranspose2dKernel<<<1, 1>>>(up1_batchnorm_1_output, up2_up_weight, up2_up_bias,
      //                 up2_convt_0_output, 2, 0);
      // ConcatKernel<<<1, 1>>>(up2_convt_0_output, inc_batchnorm_1_output, up2_concat_0_output);
      // Conv2dKernel<<<1, 1>>>(up2_concat_0_output, up2_conv_double_conv_0_weight, NULL,
      //       up2_conv_0_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(up2_conv_0_output, up2_conv_double_conv_1_weight,
      //             up2_conv_double_conv_1_bias, up2_batchnorm_0_running_mean,
      //             up2_batchnorm_0_running_var, up2_batchnorm_0_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(up2_batchnorm_0_output);
      // Conv2dKernel<<<1, 1>>>(up2_batchnorm_0_output, up2_conv_double_conv_3_weight, NULL,
      //       up2_conv_1_output, 1, 1, 1, false);
      // BatchNorm2dKernel<<<1, 1>>>(up2_conv_1_output, up2_conv_double_conv_4_weight,
      //             up2_conv_double_conv_4_bias, up2_batchnorm_1_running_mean,
      //             up2_batchnorm_1_running_var, up2_batchnorm_1_output, 1e-5, 0.1);
      // ReLUKernel<<<1, 1>>>(up2_batchnorm_1_output);

      // // outcKernel<<<grid_gen(, BLOCK_SIZE), blockDim>>>(64, 2)
      // Conv2dKernel<<<1, 1>>>(up2_batchnorm_1_output, outc_conv_weight, outc_conv_bias, output, 1,
      //       0, 1, true);