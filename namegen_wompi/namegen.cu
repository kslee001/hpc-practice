#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>

// Defined in main.cpp
extern int mpi_rank, mpi_size;

#define BLOCK_SIZE 32
#define WPT 16
#define RTS 2

#define BS 1



// cuda check
#define CHECK_CUDA(f)                                                 \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }


// // You can modify the data structure as you want
struct Tensor {
  // Pointer to data
  float *buf_cpu = nullptr;
  // gpu
  float *buf_gpu = nullptr;
  // current buf pointer
  float *buf = buf_cpu;

  // shape
  size_t ndim = 0;
  size_t shape[4];

  /* Alloc memory */  // cpu 
  Tensor(std::vector<int> shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf_cpu = (float *)malloc(n * sizeof(float));
    buf = buf_cpu;
  }

  /* Alloc memory and copy */ // cpu
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf_cpu = (float *)malloc(n * sizeof(float));
    memcpy(buf_cpu, buf_, n * sizeof(float));
    buf = buf_cpu;
  }

  ~Tensor() {
    if (buf_cpu != nullptr){
      free(buf_cpu);
    }
    if (buf_gpu != nullptr){
      CHECK_CUDA(cudaFree(buf_gpu));
    }
  }

  void set_zero() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf_cpu[i] = 0.0;
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  void gpu(){
    // allocate device memory 
    CHECK_CUDA(
      cudaMalloc(&buf_gpu, sizeof(float)*num_elem())
    );

    // copy data into device memory
    CHECK_CUDA(
      cudaMemcpy(buf_gpu, buf_cpu, sizeof(float)*num_elem(), cudaMemcpyHostToDevice)
    );
    buf = buf_gpu; // pointer direction : gpu
  }

  void cpu(){
    // CHECK_CUDA(
      cudaMemcpy(
        buf_cpu, buf_gpu, 
        sizeof(float)*num_elem(), cudaMemcpyDeviceToHost)
    // )
    ;
    buf = buf_cpu;
  }

};

dim3 grid_gen(Tensor* curTensor, int bs){
  return dim3( (curTensor->shape[0]+bs-1)/bs, (curTensor->shape[1]+bs-1)/bs );
}

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

/* Operations */



// __global__ void embeddingKernel(Tensor *input, Tensor *weight, Tensor )

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
__global__ 
void embeddingKernel(
  float* input, float*weight, float*output){
  
  int cur_row = blockIdx.y*blockDim.y + threadIdx.y;
  int cur_col = blockIdx.x*blockDim.x + threadIdx.x;

  int x = (int) input[cur_row]; // single character, 0-255
  
  if (cur_row < BS && cur_col < EMBEDDING_DIM){
      output[cur_row*EMBEDDING_DIM + cur_col] = weight[x*EMBEDDING_DIM + cur_col];
  }
}
void embeddingDevice(Tensor *input, Tensor *weight, Tensor *output) {
  // 얘네는 dimension을 x, y 순으로 준다는 점에 주의할 것 !!
  dim3 blockDim(BLOCK_SIZE, BS); // 한 블록에 쓰레드를 [0] x [1] 개 만든다.
  dim3 gridDim((EMBEDDING_DIM+BLOCK_SIZE-1)/BLOCK_SIZE, BS);   // 한 grid에 블록을 [0] x [1] 개 만든다.

  // kernel<<block개수, thread 개수>> -> kernel<<gridDim, blockDim>>
  // CHECK_CUDA(
  embeddingKernel<<<gridDim, blockDim>>>(
    input->buf, weight->buf, output->buf
  );
  // );
  cudaDeviceSynchronize();
}
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[2];
  for (size_t i = 0; i < n; i++) {
    int x = (int)input->buf[0];
    output->buf[i] = weight->buf[x * n + i];
  }
}
/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
__global__
void elemwiseAddKernel(
  float* input1, float* input2, float* output
  ){
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    output[y*BLOCK_SIZE + x] = input1[y*BLOCK_SIZE + x] + input2[y*BLOCK_SIZE + x];
}
void elemwiseAddDevice(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((sn+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  elemwiseAddKernel<<<gridDim, blockDim>>>(
    input1->buf, input2->buf, output->buf
  );
  cudaDeviceSynchronize();
}
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
__global__
void elemwiseOneminusKernel(
  float* input, float* output
  ){
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    output[y*BLOCK_SIZE + x] = 1.0 - input[y*BLOCK_SIZE+x];
}
void elemwiseOneminusDevice(Tensor *input, Tensor *output) {
  size_t sn = input->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((sn+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  elemwiseOneminusKernel<<<gridDim, blockDim>>>(
    input->buf, output->buf
  );
  cudaDeviceSynchronize();
}
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}
/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
__global__
void elemwiseMulKernel(
  float* input1, float* input2, float* output
  ){
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    output[y*BLOCK_SIZE + x] = input1[y*BLOCK_SIZE + x]*input2[y*BLOCK_SIZE + x];
}
void elemwiseMulDevice(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((sn+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  elemwiseMulKernel<<<gridDim, blockDim>>>(
    input1->buf, input2->buf, output->buf
  );
  cudaDeviceSynchronize();
}
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
__global__
void elemwiseTanhKernel(
  float* input, float* output
  ){
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    output[y*BLOCK_SIZE + x] = tanhf(input[y*BLOCK_SIZE+x]);
}
void elemwiseTanhDevice(Tensor *input, Tensor *output) {
  size_t sn = input->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((sn+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  elemwiseTanhKernel<<<gridDim, blockDim>>>(
    input->buf, output->buf
  );
  cudaDeviceSynchronize();
}
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}
/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
__global__
void elemwiseSigmoidKernel(
  float* input, float* output
  ){
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    output[y*BLOCK_SIZE + x] = 1.0/(1.0+expf(-input[y*BLOCK_SIZE+x]));
}
void elemwiseSigmoidDevice(Tensor *input, Tensor *output) {
  size_t sn = input->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((sn+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  elemwiseSigmoidKernel<<<gridDim, blockDim>>>(
    input->buf, output->buf
  );
  cudaDeviceSynchronize();
}
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}
/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
// __global__ matvecKernel(
//   float* input1, float* input2, float* output
//   const int N_, const int K_
//   ){

//   int y = blockIdx.y*BLOCK_SIZE + threadIdx.y;
//   int x = blockIdx.x*BLOCK_SIZE + threadIdx.x;



// }

// void matvecDevice(Tensor *input1, Tensor *input2, Tensor *output){
//   size_t N_ = input1->shape[0];
//   size_t K_ = input1->shape[1];

//   dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
//   dim3 gridDim((K+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE); // x, y

//   matvecKernel<<<gridDim, blockDim>>>(
//     input1->buf, input2->buf, output->buf,
//     N_, K_
//   );

// }
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[1];
  size_t K_ = input1->shape[2];
  // printf("%d, %d \n", N_, K_);
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
__global__ void matmulKernel(float *A, float *B, float *C, int M, int N, int K) {
  __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

  int global_row = blockDim.y * blockIdx.y + threadIdx.y;
  int global_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0.0f;
  int num_tiles = (K+BLOCK_SIZE-1)/BLOCK_SIZE;

  for (int tile = 0; tile < num_tiles; ++tile){
    if ( (global_row < M) && (threadIdx.x + tile*BLOCK_SIZE < K) )
      Asub[threadIdx.y][threadIdx.x] = A[global_row * K + threadIdx.x + tile * BLOCK_SIZE];
    else
      Asub[threadIdx.y][threadIdx.x] = 0.0f;

    if ( (global_col < N) && (threadIdx.y + tile * BLOCK_SIZE < K ))
      Bsub[threadIdx.y][threadIdx.x] = B[(threadIdx.y + tile * BLOCK_SIZE) * N + global_col];
    else
      Bsub[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    
    for (int k = 0; k < BLOCK_SIZE; k++)
      sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
    __syncthreads();
  }

  if (global_row < M && global_col < N){
    C[global_row * N + global_col] = sum;
  }
}
void matmulDevice(Tensor *input1, Tensor *input2, Tensor *output){
  const int M = output->shape[1];
  const int N = output->shape[2];
  const int K = input1->shape[2];
  dim3 blockDim( BLOCK_SIZE, BLOCK_SIZE, BS );
  dim3 gridDim( (N+BLOCK_SIZE-1)/BLOCK_SIZE + 1, ( M+BLOCK_SIZE-1)/BLOCK_SIZE, BS );

  matmulKernel<<<gridDim, blockDim>>>(
    input1->buf, input2->buf, output->buf, M, N, K
  );
  cudaDeviceSynchronize();
}

void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[1];
  size_t K_ = input1->shape[2];
  size_t N_ = input2->shape[2];

  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
__global__
void softmaxKernel(float* input, float* output, const int N) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate the sum of exponentials in the block
  float sum = 0.0f;
  while (i < N) {
    float x = input[i];
    sum += expf(x);
    i += blockDim.x * gridDim.x;
  }
  sdata[tid] = sum;
  __syncthreads();

  // Reduce the sum across threads in the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Calculate the output probabilities
  i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < N) {
    float x = input[i];
    output[i] = expf(x) / sdata[0];
    i += blockDim.x * gridDim.x;
  }
}
void softmaxDevice(Tensor *input, Tensor* output){
  size_t N = input->num_elem();

  dim3 blockDim(BLOCK_SIZE, BS);
  dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE, BS);
  
  softmaxKernel<<<gridDim, blockDim>>>(
    input->buf, output->buf, N
  );
  cudaDeviceSynchronize();
}
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, int rng_seed, char *parameter_fname) {

  /* Only the root process reads the parameter */
  if (mpi_rank == 0) {
    size_t parameter_binary_size = 0;
    float *parameter =
        (float *)read_binary(parameter_fname, &parameter_binary_size);

    /* Network parameters */
    character_embedding =
        new Tensor({BS, NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

    W_ir0 = new Tensor({BS, HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
    W_iz0 = new Tensor({BS, HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
    W_in0 = new Tensor({BS, HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
    W_ir1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
    W_iz1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
    W_in1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

    W_hr0 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
    W_hz0 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
    W_hn0 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
    W_hr1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
    W_hz1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
    W_hn1 = new Tensor({BS, HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

    b_ir0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET13);
    b_iz0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET14);
    b_in0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET15);
    b_ir1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET16);
    b_iz1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET17);
    b_in1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET18);

    b_hr0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET19);
    b_hz0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET20);
    b_hn0 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET21);
    b_hr1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET22);
    b_hz1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET23);
    b_hn1 = new Tensor({BS, HIDDEN_DIM}, parameter + OFFSET24);

    W_fc = new Tensor({BS, NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
    b_fc = new Tensor({BS, NUM_CHAR}, parameter + OFFSET26);

    /* input, activations, output, etc. */
    input = new Tensor({BS, 1});
    emb_out = new Tensor({BS, EMBEDDING_DIM});

    hidden0 = new Tensor({BS, HIDDEN_DIM});
    hidden1 = new Tensor({BS, HIDDEN_DIM});

    r0 = new Tensor({BS, HIDDEN_DIM});
    r1 = new Tensor({BS, HIDDEN_DIM});
    z0 = new Tensor({BS, HIDDEN_DIM});
    z1 = new Tensor({BS, HIDDEN_DIM});
    n0 = new Tensor({BS, HIDDEN_DIM});
    n1 = new Tensor({BS, HIDDEN_DIM});
    f = new Tensor({BS, NUM_CHAR});

    rtmp00 = new Tensor({BS, HIDDEN_DIM});
    rtmp01 = new Tensor({BS, HIDDEN_DIM});
    rtmp02 = new Tensor({BS, HIDDEN_DIM});
    rtmp03 = new Tensor({BS, HIDDEN_DIM});
    rtmp04 = new Tensor({BS, HIDDEN_DIM});
    rtmp10 = new Tensor({BS, HIDDEN_DIM});
    rtmp11 = new Tensor({BS, HIDDEN_DIM});
    rtmp12 = new Tensor({BS, HIDDEN_DIM});
    rtmp13 = new Tensor({BS, HIDDEN_DIM});
    rtmp14 = new Tensor({BS, HIDDEN_DIM});

    ztmp00 = new Tensor({BS, HIDDEN_DIM});
    ztmp01 = new Tensor({BS, HIDDEN_DIM});
    ztmp02 = new Tensor({BS, HIDDEN_DIM});
    ztmp03 = new Tensor({BS, HIDDEN_DIM});
    ztmp04 = new Tensor({BS, HIDDEN_DIM});
    ztmp10 = new Tensor({BS, HIDDEN_DIM});
    ztmp11 = new Tensor({BS, HIDDEN_DIM});
    ztmp12 = new Tensor({BS, HIDDEN_DIM});
    ztmp13 = new Tensor({BS, HIDDEN_DIM});
    ztmp14 = new Tensor({BS, HIDDEN_DIM});

    ntmp00 = new Tensor({BS, HIDDEN_DIM});
    ntmp01 = new Tensor({BS, HIDDEN_DIM});
    ntmp02 = new Tensor({BS, HIDDEN_DIM});
    ntmp03 = new Tensor({BS, HIDDEN_DIM});
    ntmp04 = new Tensor({BS, HIDDEN_DIM});
    ntmp05 = new Tensor({BS, HIDDEN_DIM});
    ntmp10 = new Tensor({BS, HIDDEN_DIM});
    ntmp11 = new Tensor({BS, HIDDEN_DIM});
    ntmp12 = new Tensor({BS, HIDDEN_DIM});
    ntmp13 = new Tensor({BS, HIDDEN_DIM});
    ntmp14 = new Tensor({BS, HIDDEN_DIM});
    ntmp15 = new Tensor({BS, HIDDEN_DIM});

    htmp00 = new Tensor({BS, HIDDEN_DIM});
    htmp01 = new Tensor({BS, HIDDEN_DIM});
    htmp02 = new Tensor({BS, HIDDEN_DIM});
    htmp10 = new Tensor({BS, HIDDEN_DIM});
    htmp11 = new Tensor({BS, HIDDEN_DIM});
    htmp12 = new Tensor({BS, HIDDEN_DIM});

    rfloats = new Tensor({BS, N * MAX_LEN});
    ftmp0 = new Tensor({BS, NUM_CHAR});
    char_prob = new Tensor({BS, NUM_CHAR});

    /* to device */
    // character_embedding->gpu();

    // W_ir0->gpu();
    // W_iz0->gpu();
    // W_in0->gpu();
    // W_ir1->gpu();
    // W_iz1->gpu();
    // W_in1->gpu();

    // W_hr0->gpu();
    // W_hz0->gpu();
    // W_hn0->gpu();
    // W_hr1->gpu();
    // W_hz1->gpu();
    // W_hn1->gpu();

    // b_ir0->gpu();
    // b_iz0->gpu();
    // b_in0->gpu();
    // b_ir1->gpu();
    // b_iz1->gpu();
    // b_in1->gpu();

    // b_hr0->gpu();
    // b_hz0->gpu();
    // b_hn0->gpu();
    // b_hr1->gpu();
    // b_hz1->gpu();
    // b_hn1->gpu();

    // W_fc->gpu();
    // b_fc->gpu();

    // /* input, activations, output, etc. */


    // r0->gpu();
    // r1->gpu();
    // z0->gpu();
    // z1->gpu();
    // n0->gpu();
    // n1->gpu();
    // f->gpu();

    // rtmp00->gpu();
    // rtmp01->gpu();
    // rtmp02->gpu();
    // rtmp03->gpu();
    // rtmp04->gpu();
    // rtmp10->gpu();
    // rtmp11->gpu();
    // rtmp12->gpu();
    // rtmp13->gpu();
    // rtmp14->gpu();

    // ztmp00->gpu();
    // ztmp01->gpu();
    // ztmp02->gpu();
    // ztmp03->gpu();
    // ztmp04->gpu();
    // ztmp10->gpu();
    // ztmp11->gpu();
    // ztmp12->gpu();
    // ztmp13->gpu();
    // ztmp14->gpu();

    // ntmp00->gpu();
    // ntmp01->gpu();
    // ntmp02->gpu();
    // ntmp03->gpu();
    // ntmp04->gpu();
    // ntmp05->gpu();
    // ntmp10->gpu();
    // ntmp11->gpu();
    // ntmp12->gpu();
    // ntmp13->gpu();
    // ntmp14->gpu();
    // ntmp15->gpu();

    // htmp00->gpu();
    // htmp01->gpu();
    // htmp02->gpu();
    // htmp10->gpu();
    // htmp11->gpu();
    // htmp12->gpu();

    // ftmp0->gpu();

  } 
  else {
  }
}

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  /* Only root process does the job, for now... */
  if (mpi_rank != 0)
    return;

  memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));

  /* Generate N names */
  for (int n = 0; n < N; n++) {
    /* Initialize input and hidden vector. */
    /* One hidden vector for each GRU layer */
    input->buf[0] = SOS;
    hidden0->set_zero();
    hidden1->set_zero();

    for (int l = 0; l < MAX_LEN; l++) {
      /* Embedding */
      // embedding(input, character_embedding, emb_out);

      input->gpu();
      character_embedding->gpu();
      emb_out->gpu();

      W_ir0->gpu();
      W_hr0->gpu();
      W_iz0->gpu();
      W_hz0->gpu();

      r0->gpu();
      z0->gpu();
      b_ir0->gpu();
      b_hr0->gpu();
      b_iz0->gpu();
      b_hz0->gpu();

      rtmp00->gpu();
      rtmp01->gpu();
      rtmp02->gpu();
      rtmp03->gpu();
      rtmp04->gpu();

      ztmp00->gpu();
      ztmp01->gpu();
      ztmp02->gpu();
      ztmp03->gpu();
      ztmp04->gpu();

      embeddingDevice(input, character_embedding, emb_out);
      /* First layer r */
      matmulDevice(W_ir0, emb_out, rtmp00);
      matmulDevice(W_hr0, emb_out, rtmp01);

      elemwiseAddDevice(rtmp00, b_ir0, rtmp02);
      elemwiseAddDevice(rtmp02, rtmp01, rtmp03);
      elemwiseAddDevice(rtmp03, b_hr0, rtmp04);
      elemwiseSigmoidDevice(rtmp04, r0);


      input->cpu();
      character_embedding->cpu();
      emb_out->cpu();
      
      W_ir0->cpu();
      W_hr0->cpu();
      W_iz0->cpu();
      W_hz0->cpu();
      
      r0->cpu();
      z0->cpu();
      b_ir0->cpu();
      b_hr0->cpu();
      b_iz0->cpu();
      b_hz0->cpu();

      rtmp00->cpu();
      rtmp01->cpu();
      rtmp02->cpu();
      rtmp03->cpu();
      rtmp04->cpu();

      ztmp00->cpu();
      ztmp01->cpu();
      ztmp02->cpu();
      ztmp03->cpu();
      ztmp04->cpu();

      // elemwiseSigmoidDevice(ztmp04, z0);
      /* First layer z */
      matvec(W_iz0, emb_out, ztmp00);
      matvec(W_hz0, hidden0, ztmp01);
      elemwise_add(ztmp00, b_iz0, ztmp02);
      elemwise_add(ztmp02, ztmp01, ztmp03);
      elemwise_add(ztmp03, b_hz0, ztmp04);
      elemwise_sigmoid(ztmp04, z0);

      /* First layer n */
      matvec(W_in0, emb_out, ntmp00);
      elemwise_add(ntmp00, b_in0, ntmp01);
      matvec(W_hn0, hidden0, ntmp02);
      elemwise_add(ntmp02, b_hn0, ntmp03);
      elemwise_mul(r0, ntmp03, ntmp04);
      elemwise_add(ntmp01, ntmp04, ntmp05);
      elemwise_tanh(ntmp05, n0);

      /* First layer h (hidden) */
      elemwise_oneminus(z0, htmp00);
      elemwise_mul(htmp00, n0, htmp01);
      elemwise_mul(z0, hidden0, htmp02);
      elemwise_add(htmp01, htmp02, hidden0);

      /* Second layer r */
      matvec(W_ir1, hidden0, rtmp10);
      matvec(W_hr1, hidden1, rtmp11);
      elemwise_add(rtmp10, b_ir1, rtmp12);
      elemwise_add(rtmp12, rtmp11, rtmp13);
      elemwise_add(rtmp13, b_hr1, rtmp14);
      elemwise_sigmoid(rtmp14, r1);

      /* Second layer z */
      matvec(W_iz1, hidden0, ztmp10);
      matvec(W_hz1, hidden1, ztmp11);
      elemwise_add(ztmp10, b_iz1, ztmp12);
      elemwise_add(ztmp12, ztmp11, ztmp13);
      elemwise_add(ztmp13, b_hz1, ztmp14);
      elemwise_sigmoid(ztmp14, z1);

      /* Second layer n */
      matvec(W_in1, hidden0, ntmp10);
      elemwise_add(ntmp10, b_in1, ntmp11);
      matvec(W_hn1, hidden1, ntmp12);
      elemwise_add(ntmp12, b_hn1, ntmp13);
      elemwise_mul(r1, ntmp13, ntmp14);
      elemwise_add(ntmp11, ntmp14, ntmp15);
      elemwise_tanh(ntmp15, n1);

      /* Second layer h (hidden) */
      elemwise_oneminus(z1, htmp10);
      elemwise_mul(htmp10, n1, htmp11);
      elemwise_mul(z1, hidden1, htmp12);
      elemwise_add(htmp11, htmp12, hidden1);

      /* Fully connected layer */
      // printf("here?\n");
      matvec(W_fc, hidden1, ftmp0); // ftmp0 : (BS * num_char)
      // printf("or here?");
      // W_fc->gpu();
      // hidden1->gpu();
      // ftmp0->gpu();
      // b_fc->gpu();
      // f->gpu();
      elemwise_add(ftmp0, b_fc, f);
      // W_fc->cpu();
      // hidden1->cpu();
      // ftmp0->cpu();
      // b_fc->cpu();
      // f->cpu();
      /* Softmax */
      softmax(f, char_prob);
      /* Random select */
      int selected_char = random_select(char_prob, rfloats, n * MAX_LEN + l);

      output[n * (MAX_LEN + 1) + l] = selected_char;
      input->buf[0] = selected_char;

      if (selected_char == EOS)
        break;
    }
  }
}


/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {
  if (mpi_rank == 0) {
    delete character_embedding;
    delete W_ir0;
    delete W_iz0;
    delete W_in0;
    delete W_ir1;
    delete W_iz1;
    delete W_in1;
    delete W_hr0;
    delete W_hz0;
    delete W_hn0;
    delete W_hr1;
    delete W_hz1;
    delete W_hn1;
    delete b_ir0;
    delete b_iz0;
    delete b_in0;
    delete b_ir1;
    delete b_iz1;
    delete b_in1;
    delete b_hr0;
    delete b_hz0;
    delete b_hn0;
    delete b_hr1;
    delete b_hz1;
    delete b_hn1;
    delete W_fc;
    delete b_fc;
    delete rfloats;

    delete input;
    delete emb_out;
    delete hidden0;
    delete hidden1;
    delete r0;
    delete r1;
    delete z0;
    delete z1;
    delete n0;
    delete n1;
    delete f;
    delete char_prob;
    delete rtmp00;
    delete rtmp01;
    delete rtmp02;
    delete rtmp03;
    delete rtmp04;
    delete rtmp10;
    delete rtmp11;
    delete rtmp12;
    delete rtmp13;
    delete rtmp14;
    delete ztmp00;
    delete ztmp01;
    delete ztmp02;
    delete ztmp03;
    delete ztmp04;
    delete ztmp10;
    delete ztmp11;
    delete ztmp12;
    delete ztmp13;
    delete ztmp14;
    delete ntmp00;
    delete ntmp01;
    delete ntmp02;
    delete ntmp03;
    delete ntmp04;
    delete ntmp05;
    delete ntmp10;
    delete ntmp11;
    delete ntmp12;
    delete ntmp13;
    delete ntmp14;
    delete ntmp15;
    delete htmp00;
    delete htmp01;
    delete htmp02;
    delete htmp10;
    delete htmp11;
    delete htmp12;
    delete ftmp0;
  }
}