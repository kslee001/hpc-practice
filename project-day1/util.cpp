#include "util.h"

#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

// Defined in main.cpp
extern int N;
extern char *parameter_fname;
extern char *output_fname;
extern char *input_fname;

void read_binary(void *dst, const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  CHECK_ERROR(f != NULL, "Failed to read %s", filename);
  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  size_t ret = fread(dst, 1, size_, f);
  fclose(f);
  CHECK_ERROR(size_ == ret, "Failed to read %ld bytes from %s", size_,
              filename);
  if (size != NULL) *size = (size_t)(size_ / 4);  // float
}

void *read_binary(const char *filename, size_t *size) {
  size_t size_;
  FILE *f = fopen(filename, "rb");
  CHECK_ERROR(f != NULL, "Failed to read %s", filename);
  fseek(f, 0, SEEK_END);
  size_ = ftell(f);
  rewind(f);
  void *buf = malloc(size_);
  size_t ret = fread(buf, 1, size_, f);
  fclose(f);
  CHECK_ERROR(size_ == ret, "Failed to read %ld bytes from %s", size_,
              filename);
  if (size != NULL) *size = (size_t)(size_ / 4);  // float
  return buf;
}

void write_binary(void *dst, const char *filename, size_t size) {
  FILE *output_fp = (FILE *) fopen(output_fname, "wb");
  fwrite(dst, sizeof(float), size, output_fp);
  fclose(output_fp);
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_usage_exit(int argc, char **argv) {
  printf("Usage %s [parameter bin] [input bin] [output] [N]\n", argv[0]);
  printf("  parameter bin: File containing DNN parameters\n");
  printf("  input bin : File containing input images\n");
  printf("  output: File to write results\n");
  printf("  N: Number of images to mask\n");
  EXIT(0);
}

void check_and_parse_args(int argc, char **argv) {
  if (argc != 5) print_usage_exit(argc, argv);

  int c;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
      case 'h': break;
      default: print_usage_exit(argc, argv);
    }
  }

  parameter_fname = argv[1];
  input_fname = argv[2];
  output_fname = argv[3];
  N = atoi(argv[4]);
}

void print_model() {
  printf("\n Model : U-Net\n");
  printf(
      "------------------------------------------------------------------\n");
  printf(" Automatically identify the boundaries of the images in input.bin\n");
  printf(
      "==================================================================\n");
  printf(" Number of input images : %d\n", N);
  printf(" Parameter file : %s\n", parameter_fname);
  printf(" Input file : %s\n", input_fname);
  printf(" Output file to write results : %s\n", output_fname);
  printf(
      "==================================================================\n");
}
