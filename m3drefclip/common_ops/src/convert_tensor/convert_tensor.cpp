//
// Created by Yiming Zhang on 5/23/23.
//


#include "convert_tensor.h"
void convertSparseTensorToDense(const at::Tensor &sparseTensor, const at::Tensor &offsetTensor, const at::Tensor &denseTensor) {
    const float *sparse = sparseTensor.data_ptr<float>();
    const int16_t *offset = offsetTensor.data_ptr<int16_t>();
    float *dense = denseTensor.data_ptr<float>();

    convertSparseTensorToDenseCuda(sparse, offset, dense, denseTensor.sizes()[0], denseTensor.sizes()[1], denseTensor.sizes()[2]);
}