#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <math.h>
#define TS 32
#define WPT 16
#define RTS 2

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
int num_devices = 0;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {

  __shared__ float Asub[TS][TS];
  __shared__ float Bsub[TS][TS];

  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const int gi = blockIdx.x * TS + i;
  const int gj = blockIdx.y * TS + j;

  float acc[WPT];
  for (int w = 0; w < WPT; w++) {
    acc[w] = 0.0f;
  }

  int nT = (K + TS - 1) /TS;
  for (int t = 0; t < nT; t++){
    for (int w = 0; w < WPT; w++) {
      const int ti = TS * t + i;
      const int tj = TS * t + j;

      if ((gi + w * RTS) < M && tj < K) {
        Asub[i + w * RTS][j] = A[(gi + w * RTS) * K + tj];
      }
      else {
        Asub[i + w * RTS][j] = 0.0f;
      }
      if ((ti + w * RTS) < K && gj < N){
        Bsub[i + w * RTS][j] = B[(ti + w * RTS) * N + gj];
      }
      else {
        Bsub[i + w * RTS][j] = 0.0f;
      }
    }
    __syncthreads();
    for (int k = 0; k < TS; k++) {
      for (int w = 0; w < WPT; w++) {
        acc[w] += Asub[i + w * RTS][k] * Bsub[k][j];
      }
    }
    __syncthreads();
  }

  for (int w = 0; w < WPT; w++) {
    if ((gi + w * RTS) < M && gj < N) {
      C[(gi + w * RTS) * N + gj] = acc[w];
    }
  }
}

static int mpi_rank, mpi_world_size;

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];
static int hostBegin[4], hostEnd[4];
static MPI_Status status;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // send & recv data (root -> CPUs)
  if (mpi_rank==0) {
    for (int i=1; i< mpi_world_size; i++) {
      MPI_Send((float*) A+hostBegin[i]*K, (hostEnd[i]-hostBegin[i])*K, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
      MPI_Send((float*) B, K*N, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
    }
  } else {
      MPI_Recv((float*) A+hostBegin[mpi_rank]*K, (hostEnd[mpi_rank]-hostBegin[mpi_rank])*K, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Recv((float*) B, K*N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Upload A and B matrix to every GPU
  for (int i = 0; i < num_devices; i++) { //a_d (global) <- A (cpu)
    CUDA_CALL(cudaMemcpy(a_d[i], A + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    int Msub = Mend[i] - Mbegin[i];
    dim3 blockDim(TS/WPT, TS, 1);
    dim3 gridDim((Msub + TS - 1) / TS, (N + TS - 1) / TS, 1);
    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(C + Mbegin[i] * N, c_d[i],
                        (Mend[i] - Mbegin[i]) * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  }
 
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    for (int i=1; i < mpi_world_size; i++) {
      MPI_Recv((float*) C+hostBegin[i]*N, (hostEnd[i]-hostBegin[i])*N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
    }
  } else {
    MPI_Send((float*) C+hostBegin[mpi_rank]*N, (hostEnd[mpi_rank]-hostBegin[mpi_rank])*N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
  // setting Mbegin & Mend for each host(cpu)
  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  for (int i=0; i<mpi_world_size; ++i) {
    hostBegin[i] =  (M/mpi_world_size)*i;
    hostEnd[i] = (M/mpi_world_size)*(i+1);
  }
  hostEnd[mpi_world_size-1] = M;

  int portion = hostEnd[mpi_rank]-hostBegin[mpi_rank];
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = hostBegin[mpi_rank] + (portion / num_devices) * i;
    Mend[i] = hostBegin[mpi_rank] + (portion / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = hostEnd[mpi_rank];
  MPI_Barrier(MPI_COMM_WORLD);

  // Only root process do something
  if (mpi_rank == 0) {
    printf("Using %d devices\n", num_devices);
    for (int i = 0; i < num_devices; i++) {
      cudaDeviceProp prop;
      CUDA_CALL(cudaGetDeviceProperties(&prop, i));

      // Try printing more detailed information here
      printf("GPU %d: %s\n", i, prop.name);
    }

    if (num_devices <= 0) {
      printf("No CUDA device found. Aborting\n");
      exit(1);
    }
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }  
}

void matmul_finalize() {

  // Only root process do something
  if (mpi_rank == 0) {
    // Free all GPU memory
    for (int i = 0; i < num_devices; i++) {
      CUDA_CALL(cudaFree(a_d[i]));
      CUDA_CALL(cudaFree(b_d[i]));
      CUDA_CALL(cudaFree(c_d[i]));
    }
  }
}