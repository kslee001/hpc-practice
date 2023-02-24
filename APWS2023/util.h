#pragma once

#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

/* Useful macros */
#define EXIT(status) \
  do { exit(status); } while (0)

#define CHECK_ERROR(cond, fmt, ...)    \
  do {                                 \
    if (!(cond)) {                     \
      printf(fmt "\n", ##__VA_ARGS__); \
      EXIT(EXIT_FAILURE);              \
    }                                  \
  } while (false)

void print_usage_exit(int argc, char **argv);
void check_and_parse_args(int argc, char **argv);
double get_time();
void read_binary(void *dst, const char *filename, size_t *size);
void *read_binary(const char *filename, size_t *size);
void write_binary(void *dst, const char *filename, size_t size);
void print_first_few_result(float *output, int print_max, double elapsed_time);
void print_model();