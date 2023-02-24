#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "uNet.h"
#include "util.h"

int WARMUP = 0;
int MEASURE = 1;

// Global variables
int N = 1;
char *parameter_fname;
char *output_fname;
char *input_fname;

/*
Execute Parameters
  argv[1] = unet model parameters path
  argv[2] = input image path
  argv[3] = output save path
  argv[4] = the number of images to inference

  // optional
  argv[5] = warming up count
  argv[6] = performance measuring count

*/
int main(int argc, char **argv) {
  check_and_parse_args(argc, argv);
  print_model();

  // Initialize model
  uNet_initialize(N, parameter_fname);

  Tensor *input = new Tensor({N, 3, 128, 191});
  Tensor *output = new Tensor({N, 2, 128, 191});

  size_t input_size = 0;
  read_binary((void *) input->buf, input_fname, &input_size);

  if (argc > 5){
    WARMUP = atoi(argv[5]);
    MEASURE = atoi(argv[6]);
  }

  printf(" process %d image(s)...", N);
  printf(" Warm up [%d], Performance measure [%d]\n",WARMUP, MEASURE);
  fflush(stdout);

  // warming up
  printf("\nWarmimg up.");
  fflush(stdout);
  for (int i = 0; i < WARMUP; ++i) {
    uNet(input, output, N);
    fflush(stdout);
    printf(".");
  }
  cudaDeviceSynchronize();
  printf("\n");

  // performance measure
  printf("\nProcess.");
  double uNet_st = get_time();
  for (int j = 0; j < MEASURE; ++j) {
    uNet(input, output, N);
    printf(".");
  }
  cudaDeviceSynchronize();
  double uNet_en = get_time();
  printf("\n");
  double elapsed_time = uNet_en - uNet_st;
  elapsed_time = elapsed_time / MEASURE;
  printf("%lfsec (%lf img/sec)\n", elapsed_time, N / elapsed_time);

  write_binary((void *) output->buf, output_fname, (size_t)(N * 2 * 128 * 191));

  printf("Writing final result to %s ...", output_fname);
  fflush(stdout);

  printf("Done!\n\n");

  // Finalize program
  uNet_finalize();
}