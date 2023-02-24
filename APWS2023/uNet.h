#pragma once

#include "tensor.h"

// Model parameters
#define PARAMETER_FILE_SIZE 45663232
#define NUM_IMAGES 256

#define OFFSET0 0
#define OFFSET1 (OFFSET0 + 64 * 3 * 3 * 3)
#define OFFSET2 (OFFSET1 + 64)
#define OFFSET3 (OFFSET2 + 64)
#define OFFSET4 (OFFSET3 + 64 * 64 * 3 * 3)
#define OFFSET5 (OFFSET4 + 64)
#define OFFSET6 (OFFSET5 + 64)
#define OFFSET7 (OFFSET6 + 128 * 64 * 3 * 3)
#define OFFSET8 (OFFSET7 + 128)
#define OFFSET9 (OFFSET8 + 128)
#define OFFSET10 (OFFSET9 + 128 * 128 * 3 * 3)
#define OFFSET11 (OFFSET10 + 128)
#define OFFSET12 (OFFSET11 + 128)
#define OFFSET13 (OFFSET12 + 256 * 128 * 3 * 3)
#define OFFSET14 (OFFSET13 + 256)
#define OFFSET15 (OFFSET14 + 256)
#define OFFSET16 (OFFSET15 + 256 * 256 * 3 * 3)
#define OFFSET17 (OFFSET16 + 256)
#define OFFSET18 (OFFSET17 + 256)
#define OFFSET19 (OFFSET18 + 256 * 128 * 2 * 2)
#define OFFSET20 (OFFSET19 + 128)
#define OFFSET21 (OFFSET20 + 128 * 256 * 3 * 3)
#define OFFSET22 (OFFSET21 + 128)
#define OFFSET23 (OFFSET22 + 128)
#define OFFSET24 (OFFSET23 + 128 * 128 * 3 * 3)
#define OFFSET25 (OFFSET24 + 128)
#define OFFSET26 (OFFSET25 + 128)
#define OFFSET27 (OFFSET26 + 128 * 64 * 2 * 2)
#define OFFSET28 (OFFSET27 + 64)
#define OFFSET29 (OFFSET28 + 64 * 128 * 3 * 3)
#define OFFSET30 (OFFSET29 + 64)
#define OFFSET31 (OFFSET30 + 64)
#define OFFSET32 (OFFSET31 + 64 * 64 * 3 * 3)
#define OFFSET33 (OFFSET32 + 64)
#define OFFSET34 (OFFSET33 + 64)
#define OFFSET35 (OFFSET34 + 2 * 64 * 1 * 1)
#define OFFSET36 (OFFSET35 + 2)
#define OFFSET37 (OFFSET36 + 64)
#define OFFSET38 (OFFSET37 + 64)
#define OFFSET39 (OFFSET38 + 64)
#define OFFSET40 (OFFSET39 + 64)
#define OFFSET41 (OFFSET40 + 128)
#define OFFSET42 (OFFSET41 + 128)
#define OFFSET43 (OFFSET42 + 128)
#define OFFSET44 (OFFSET43 + 128)
#define OFFSET45 (OFFSET44 + 256)
#define OFFSET46 (OFFSET45 + 256)
#define OFFSET47 (OFFSET46 + 256)
#define OFFSET48 (OFFSET47 + 256)
#define OFFSET49 (OFFSET48 + 128)
#define OFFSET50 (OFFSET49 + 128)
#define OFFSET51 (OFFSET50 + 128)
#define OFFSET52 (OFFSET51 + 128)
#define OFFSET53 (OFFSET52 + 64)
#define OFFSET54 (OFFSET53 + 64)
#define OFFSET55 (OFFSET54 + 64)
#define OFFSET56 (OFFSET55 + 64)

void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias);
void ReLU(Tensor *inout);
void BatchNorm2d(Tensor *input, Tensor *gamma, Tensor *beta,
                 Tensor *running_mean, Tensor *running_var, Tensor *output,
                 const float eps, const float momentum);
void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *bias,
                     Tensor *output, int stride, int pad);
float max4(float, float, float, float);
void MaxPool2d(Tensor *input, Tensor *output);
void Concat(Tensor *input1, Tensor *input2, Tensor *output);
void uNet_initialize(int, char *);
void uNet(Tensor *, Tensor *, int);
void uNet_finalize();