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
  Tensor *input = new Tensor({1, 3, 128, 191});   // input image tensor
  Tensor *output = new Tensor({1, 2, 128, 191});  // output image tensor (binary : segmentation)

  for (int idx = 0; idx < N; ++idx) {
    memcpy(input->buf, inputN->buf + (idx * 1 * 3 * 128 * 191),
           sizeof(float) * 1 * 3 * 128 * 191);

    // inc(n_channels, 64)
    Conv2d(input, inc_double_conv_0_weight, NULL, inc_conv_0_output, 1, 1, 1, // input, (weight, bias,) ouput,   ( stride pad dilatation ) , bias
           false);
    BatchNorm2d(inc_conv_0_output, inc_double_conv_1_weight,
                inc_double_conv_1_bias, inc_batchnorm_0_running_mean,
                inc_batchnorm_0_running_var, inc_batchnorm_0_output, 1e-5, 0.1);  
    ReLU(inc_batchnorm_0_output);
    Conv2d(inc_batchnorm_0_output, inc_double_conv_3_weight, NULL,
           inc_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(inc_conv_1_output, inc_double_conv_4_weight,
                inc_double_conv_4_bias, inc_batchnorm_1_running_mean,
                inc_batchnorm_1_running_var, inc_batchnorm_1_output, 1e-5, 0.1);
    ReLU(inc_batchnorm_1_output);

    // down1(64, 128)
    MaxPool2d(inc_batchnorm_1_output, down1_maxpool2d_0_output);
    Conv2d(down1_maxpool2d_0_output, down1_maxpool_conv_1_double_conv_0_weight,
           NULL, down1_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(down1_conv_0_output, down1_maxpool_conv_1_double_conv_1_weight,
                down1_maxpool_conv_1_double_conv_1_bias,
                down1_batchnorm_0_running_mean, down1_batchnorm_0_running_var,
                down1_batchnorm_0_output, 1e-5, 0.1);
    ReLU(down1_batchnorm_0_output);
    Conv2d(down1_batchnorm_0_output, down1_maxpool_conv_1_double_conv_3_weight,
           NULL, down1_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(down1_conv_1_output, down1_maxpool_conv_1_double_conv_4_weight,
                down1_maxpool_conv_1_double_conv_4_bias,
                down1_batchnorm_1_running_mean, down1_batchnorm_1_running_var,
                down1_batchnorm_1_output, 1e-5, 0.1);
    ReLU(down1_batchnorm_1_output);

    // down2(128, 256)
    MaxPool2d(down1_batchnorm_1_output, down2_maxpool2d_0_output);
    Conv2d(down2_maxpool2d_0_output, down2_maxpool_conv_1_double_conv_0_weight,
           NULL, down2_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(down2_conv_0_output, down2_maxpool_conv_1_double_conv_1_weight,
                down2_maxpool_conv_1_double_conv_1_bias,
                down2_batchnorm_0_running_mean, down2_batchnorm_0_running_var,
                down2_batchnorm_0_output, 1e-5, 0.1);
    ReLU(down2_batchnorm_0_output);
    Conv2d(down2_batchnorm_0_output, down2_maxpool_conv_1_double_conv_3_weight,
           NULL, down2_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(down2_conv_1_output, down2_maxpool_conv_1_double_conv_4_weight,
                down2_maxpool_conv_1_double_conv_4_bias,
                down2_batchnorm_1_running_mean, down2_batchnorm_1_running_var,
                down2_batchnorm_1_output, 1e-5, 0.1);
    ReLU(down2_batchnorm_1_output);

    // up1(256, 128), (up2_concat_0_output, down1_batchnorm_1_output)
    ConvTranspose2d(down2_batchnorm_1_output, up1_up_weight, up1_up_bias,
                    up1_convt_0_output, 2, 0);
    Concat(up1_convt_0_output, down1_batchnorm_1_output, up1_concat_0_output);
    Conv2d(up1_concat_0_output, up1_conv_double_conv_0_weight, NULL,
           up1_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(up1_conv_0_output, up1_conv_double_conv_1_weight,
                up1_conv_double_conv_1_bias, up1_batchnorm_0_running_mean,
                up1_batchnorm_0_running_var, up1_batchnorm_0_output, 1e-5, 0.1);
    ReLU(up1_batchnorm_0_output);
    Conv2d(up1_batchnorm_0_output, up1_conv_double_conv_3_weight, NULL,
           up1_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(up1_conv_1_output, up1_conv_double_conv_4_weight,
                up1_conv_double_conv_4_bias, up1_batchnorm_1_running_mean,
                up1_batchnorm_1_running_var, up1_batchnorm_1_output, 1e-5, 0.1);
    ReLU(up1_batchnorm_1_output);

    // up2(128, 64), (up1_concat_0_output, inc_batchnorm_1_output)
    ConvTranspose2d(up1_batchnorm_1_output, up2_up_weight, up2_up_bias,
                    up2_convt_0_output, 2, 0);
    Concat(up2_convt_0_output, inc_batchnorm_1_output, up2_concat_0_output);
    Conv2d(up2_concat_0_output, up2_conv_double_conv_0_weight, NULL,
           up2_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(up2_conv_0_output, up2_conv_double_conv_1_weight,
                up2_conv_double_conv_1_bias, up2_batchnorm_0_running_mean,
                up2_batchnorm_0_running_var, up2_batchnorm_0_output, 1e-5, 0.1);
    ReLU(up2_batchnorm_0_output);
    Conv2d(up2_batchnorm_0_output, up2_conv_double_conv_3_weight, NULL,
           up2_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(up2_conv_1_output, up2_conv_double_conv_4_weight,
                up2_conv_double_conv_4_bias, up2_batchnorm_1_running_mean,
                up2_batchnorm_1_running_var, up2_batchnorm_1_output, 1e-5, 0.1);
    ReLU(up2_batchnorm_1_output);

    // outc(64, 2)
    Conv2d(up2_batchnorm_1_output, outc_conv_weight, outc_conv_bias, output, 1,
           0, 1, true);

    memcpy(outputN->buf + (idx * 1 * 2 * 128 * 191), output->buf,
           sizeof(float) * (1 * 2 * 128 * 191));
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
  printf("Kernel Size : %d  filter Width : %d   filter Height: %d   \n",K, R, S);
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
  for (int k = 0; k < K; ++k) {              // filter's channel
    for (int oh = 0; oh < OH; ++oh) {        // output hieght
      for (int ow = 0; ow < OW; ++ow) {      // output width
        float o = has_bias ? bias->buf[k] : 0;  
        for (int c = 0; c < C; ++c) {        // input's channel
          for (int r = 0; r < R; ++r) {      // filter's hieght
            for (int s = 0; s < S; ++s) {    // filter's width
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
  inc_conv_0_output = new Tensor({1, 64, 128, 191});
  inc_batchnorm_0_output = new Tensor({1, 64, 128, 191});
  inc_conv_1_output = new Tensor({1, 64, 128, 191});
  inc_batchnorm_1_output = new Tensor({1, 64, 128, 191});

  down1_maxpool2d_0_output = new Tensor({1, 64, 64, 95});
  down1_conv_0_output = new Tensor({1, 128, 64, 95});
  down1_batchnorm_0_output = new Tensor({1, 128, 64, 95});
  down1_conv_1_output = new Tensor({1, 128, 64, 95});
  down1_batchnorm_1_output = new Tensor({1, 128, 64, 95});

  down2_maxpool2d_0_output = new Tensor({1, 128, 32, 47});
  down2_conv_0_output = new Tensor({1, 256, 32, 47});
  down2_batchnorm_0_output = new Tensor({1, 256, 32, 47});
  down2_conv_1_output = new Tensor({1, 256, 32, 47});
  down2_batchnorm_1_output = new Tensor({1, 256, 32, 47});

  up1_convt_0_output = new Tensor({1, 128, 64, 94});
  up1_concat_0_output = new Tensor({1, 256, 64, 95});
  up1_conv_0_output = new Tensor({1, 128, 64, 95});
  up1_batchnorm_0_output = new Tensor({1, 128, 64, 95});
  up1_conv_1_output = new Tensor({1, 128, 64, 95});
  up1_batchnorm_1_output = new Tensor({1, 128, 64, 95});

  up2_convt_0_output = new Tensor({1, 64, 128, 190});
  up2_concat_0_output = new Tensor({1, 128, 128, 191});
  up2_conv_0_output = new Tensor({1, 64, 128, 191});
  up2_batchnorm_0_output = new Tensor({1, 64, 128, 191});
  up2_conv_1_output = new Tensor({1, 64, 128, 191});
  up2_batchnorm_1_output = new Tensor({1, 64, 128, 191});
  outc_conv_0_output = new Tensor({1, 2, 128, 191});
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
