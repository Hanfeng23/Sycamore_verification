/*  
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../../einsum.h"

template<>
struct CuTensorTypeTraits<at::Half> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<at::BFloat16> {
  static const cudaDataType_t cudaType = CUDA_R_16BF;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<float>> {
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef c10::complex<float> ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<double>> {
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef c10::complex<double> ScalarType;
};

torch::Tensor einsum(
    std::string subscripts,
    torch::Tensor opt_plan,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  // printf("here torch::Tensor einsum\n");
  at::Tensor output_tensor;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }

    output_tensor = torch::empty(myEinsum.getOutputShape(), input_0.options());

    size_t worksize = myEinsum.getWorksize();
    at::Tensor workspace = at::empty({static_cast<long int>(worksize)}, at::CUDA(at::kByte));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                opt_plan.data_ptr<int32_t>(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                worksize,
                                stream);

    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return output_tensor;
}

void einsum_gemm(torch::Tensor C,
                 torch::Tensor A,
                 torch::Tensor B,
                 size_t m,
                 size_t n,
                 size_t k,
                 size_t l = 1,
                 torch::Tensor batch_i = at::empty({0}),
                 torch::Tensor batch_j = at::empty({0}),
                 int use_cutlass = 1)
{
  // printf("in einsum_gemm, m = %d, n = %d, k = %d, l = %d\n", m, n, k, l);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      A.scalar_type(),
      "einsum_gemm_",
      [&]
      {
        cuComplex *APtr = (cuComplex *)(A.data_ptr<c10::complex<float>>());
        cuComplex *BPtr = (cuComplex *)(B.data_ptr<c10::complex<float>>());
        cuComplex *CPtr = (cuComplex *)(C.data_ptr<c10::complex<float>>());

        if (l > 1)
        {
          const long int worksize = sizeof(cuComplex *) * l * 3ULL;
          at::Tensor h_workspace = at::empty({worksize}, at::CPU(at::kByte));
          cuComplex **h_As = (cuComplex **)(h_workspace.data_ptr<uint8_t>());
          cuComplex **h_Bs = h_As + l;
          cuComplex **h_Cs = h_Bs + l;

          auto batch_i_ptr = batch_i.accessor<int64_t, 1>();
          auto batch_j_ptr = batch_j.accessor<int64_t, 1>();
          const size_t mk = m * k;
          const size_t kn = k * n;
          const size_t mn = m * n;
          for (int i = 0; i < l; ++i)
          {
            h_As[i] = APtr + mk * batch_i_ptr[i];
            h_Bs[i] = BPtr + kn * batch_j_ptr[i];
            h_Cs[i] = CPtr + mn * i;
          }

          at::Tensor d_workspace = at::empty({worksize}, at::CUDA(at::kByte));
          cuComplex **d_As = (cuComplex **)(d_workspace.data_ptr<uint8_t>());
          cuComplex **d_Bs = d_As + l;
          cuComplex **d_Cs = d_Bs + l;

          CHECK_CALL(cudaMemcpy((cuComplex **)(d_workspace.data_ptr<uint8_t>()),
                                (cuComplex **)(h_workspace.data_ptr<uint8_t>()),
                                worksize,
                                cudaMemcpyHostToDevice),
                     cudaSuccess);

          einsum_gemm_execute(m,
                              n,
                              k,
                              (void *)d_As,
                              (void *)d_Bs,
                              (void *)d_Cs,
                              l,
                              (bool)use_cutlass);
        }
        else
        {
          einsum_gemm_execute(m, n, k, APtr, BPtr, CPtr, 1, (bool)use_cutlass);
        }
      });
}

void einsum_gemm_execute(const int m,
                         const int n,
                         const int k,
                         const void *A,
                         const void *B,
                         void *C,
                         const int batch_count,
                         const bool useCutlass)
{
  // GPUTimer timer;
  // timer.start();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (useCutlass)
  {
    printf("cutlass is not supported in einsum_gemm.\n");
    
  }
  else  // cublas
  {
    // swap A and B since input and output matrices are all row-majored
    const int lda = k;
    const int ldb = k;
    const int ldc = n;
    if (batch_count > 1)
    {
      CHECK_CALL(cublasGemmBatchedEx(*GetCuBlasHandle(),
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)n,
                                     (int)m,
                                     (int)k,
                                     &alpha,
                                     (void **)B,
                                     CUDA_C_32F,
                                     (int)ldb,
                                     (void **)A,
                                     CUDA_C_32F,
                                     (int)lda,
                                     &beta,
                                     (void **)C,
                                     CUDA_C_32F,
                                     (int)ldc,
                                     (int)batch_count,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 CUBLAS_STATUS_SUCCESS);
    }
    else
    {
      CHECK_CALL(cublasGemmEx(*GetCuBlasHandle(),
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              (int)n,
                              (int)m,
                              (int)k,
                              &alpha,
                              (void *)B,
                              CUDA_C_32F,
                              (int)ldb,
                              (void *)A,
                              CUDA_C_32F,
                              (int)lda,
                              &beta,
                              (void *)C,
                              CUDA_C_32F,
                              (int)ldc,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 CUBLAS_STATUS_SUCCESS);
    }
  }
  CHECK_CALL(cudaDeviceSynchronize(), cudaSuccess);
  // auto time = timer.seconds();
  // printf("gemm: %1.6f ms\n", time * 1000);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("einsum", &einsum, "Einsum");
  m.def("einsum_gemm", &einsum_gemm, "Einsum_Gemm");
}
