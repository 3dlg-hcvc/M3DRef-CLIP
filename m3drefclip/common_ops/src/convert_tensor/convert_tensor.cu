#include "convert_tensor.h"


__global__ void convertSparseTensorToDenseCudaKernel(const float *sparse, const int16_t *offset, float *dense, const int maxNum) {

    const int currentBatch = blockIdx.x / maxNum;
    const int16_t currentOffset = (int16_t)(blockIdx.x % maxNum);
    const int16_t &startIdx = offset[currentBatch];
    const int16_t &endIdx = offset[currentBatch + 1];
    if (currentOffset >= endIdx - startIdx) {
        return;
    }
    dense[blockIdx.x * blockDim.x + threadIdx.x] = sparse[(startIdx + currentOffset) * blockDim.x + threadIdx.x];
}

void convertSparseTensorToDenseCuda(const float *sparse, const int16_t *offset, float *dense, const int &batchSize, const int &maxNum, const int &dim) {
    convertSparseTensorToDenseCudaKernel<<<batchSize * maxNum, dim>>>(sparse, offset, dense, maxNum);
}