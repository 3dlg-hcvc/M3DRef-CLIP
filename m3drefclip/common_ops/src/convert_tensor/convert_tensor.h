//
// Created by Yiming Zhang on 5/23/23.
//

#ifndef CONVERT_TENSOR_H
#define CONVERT_TENSOR_H
#include <torch/serialize/tensor.h>

void convertSparseTensorToDense(const at::Tensor &sparseTensor, const at::Tensor &offsetTensor, const at::Tensor &denseTensor);

void convertSparseTensorToDenseCuda(const float *sparse, const int16_t *offset, float *dense, const int &batchSize, const int &maxNum, const int &dim);


#endif //CONVERT_TENSOR_H
