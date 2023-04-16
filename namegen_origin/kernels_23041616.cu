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
  // if(output->is_offloading()) printf("output tensor offloaded \n");

  if (input->is_offloading() 
  && weight->is_offloading() 
  && output->is_offloading()){
    embeddingDevice(input, weight, output);
  }
  else{
    size_t n = weight->shape[2];
    for (size_t i = 0; i < n; i++) {
      int x = (int)input->buf[0];
      output->buf[i] = weight->buf[x * n + i];
    }
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
  if(input1->is_offloading() 
  && input2->is_offloading()
  && output->is_offloading()){
    elemwiseAddDevice(input1, input2, output);
  }
  else{
    size_t sn = input1->num_elem();
    for (size_t i = 0; i < sn; i++) {
      output->buf[i] = input1->buf[i] + input2->buf[i];
      }
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
  if (input->is_offloading() && output->is_offloading()){
    elemwiseOneminusDevice(input, output);    
  }
  else{
    size_t n = input->num_elem();
    for (size_t i = 0; i < n; i++) {
      float x = input->buf[i];
      output->buf[i] = 1.0 - x;
    }
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
  if (input1->is_offloading() 
  && input2->is_offloading() 
  && output->is_offloading()){
    elemwiseMulDevice(input1, input2, output);
  }
  else{
    size_t sn = input1->num_elem();
    for (size_t i = 0; i < sn; i++) {
      output->buf[i] = input1->buf[i] * input2->buf[i];
    }
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
  if (input->is_offloading() && output->is_offloading()){
    elemwiseTanhDevice(input, output);
  }
  else{
    size_t n = input->num_elem();
    for (size_t i = 0; i < n; i++) {
      float x = input->buf[i];
      output->buf[i] = tanhf(x);
    }
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
  if (input->is_offloading() && output->is_offloading()){
    elemwiseSigmoidDevice(input, output);
  }
  else{
    size_t n = input->num_elem();
    for (size_t i = 0; i < n; i++) {
      float x = input->buf[i];
      output->buf[i] = 1.0 / (1.0 + expf(-x));
    }
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
__global__ void matmulKernel2(float *A, float *B, float *C, int M, int N, int K) {
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

__global__
void matmulKernel(
  float* input1, float* input2, float* output, 
  const int M_, const int K_, const int N_
  ){

    for (size_t i = 0; i < M_; i++) {
      for (size_t j = 0; j < N_; j++) {
        float c = 0.0;
        for (size_t k = 0; k < K_; k++) {
          c += input1[i * K_ + k] * input2[k * N_ + j];
        }
        output[i * N_ + j] = c;
      }
    }
}
void matmulDevice(Tensor *input1, Tensor *input2, Tensor *output){
  size_t M_ = input1->shape[1];
  size_t K_ = input1->shape[2];
  size_t N_ = input2->shape[2];
  dim3 blockDim( BLOCK_SIZE, BLOCK_SIZE, BS );
  dim3 gridDim( (N_+BLOCK_SIZE-1)/BLOCK_SIZE + 1, ( M_+BLOCK_SIZE-1)/BLOCK_SIZE, BS );

  matmulKernel<<<gridDim, blockDim>>>(
    input1->buf, input2->buf, output->buf, M_, K_, N_
  );
  cudaDeviceSynchronize();
}

void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[1];
  size_t K_ = input1->shape[2];
  size_t N_ = input2->shape[2];

  if (input1->is_offloading() 
  && input2->is_offloading()
  && output->is_offloading()){
    matmulDevice(input1, input2, output);
  }
  else{

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
  if (input->is_offloading() && output->is_offloading()){
    softmaxDevice(input, output);
  }
  else{
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
}
