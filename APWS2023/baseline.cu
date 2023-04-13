/*
baseline

layers
- 3x3 conv
- 1x1 conv
- 2x2 maxpool
- 2x2 up conv
- relu

*/


__global__ void convolution2D(float* input, float* output, float* kernel, int inputWidth, int inputHeight, int kernelSize)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int kWidth = kernelSize / 2;

    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    float sum = 0.0;
    for (int i = -kWidth; i <= kWidth; i++) {
        for (int j = -kWidth; j <= kWidth; j++) {
            int idx = (y + i) * inputWidth + (x + j);
            int kIdx = (i + kWidth) * kernelSize + (j + kWidth);
            if (x + j >= 0 && x + j < inputWidth && y + i >= 0 && y + i < inputHeight) {
                sum += input[idx] * kernel[kIdx];
            }
        }
    }

    output[y * inputWidth + x] = sum;
}

int main()
{
    int inputWidth = 512;
    int inputHeight = 512;
    int kernelSize = 5;
    int outputWidth = inputWidth - kernelSize + 1;
    int outputHeight = inputHeight - kernelSize + 1;

    float* input = new float[inputWidth * inputHeight];
    float* output = new float[outputWidth * outputHeight];
    float* kernel = new float[kernelSize * kernelSize];

    // Initialize input, output, and kernel data

    float* d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&d_output, outputWidth * outputHeight * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(d_input, input, inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, (outputHeight + blockDim.y - 1) / blockDim.y);

    convolution2D<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, inputWidth, inputHeight, kernelSize);

    cudaMemcpy(output, d_output, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // Use the output data as needed

    return 0;
}
