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

#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "cutensor.h"

static u_int32_t contraction_count = 0;

#define HANDLE_ERROR(x)                         \
  {                                             \
    const auto err = x;                         \
    if (err == CUTENSOR_STATUS_NOT_SUPPORTED)   \
    {                                           \
      return false;                             \
    }                                           \
    if (err != CUTENSOR_STATUS_SUCCESS)         \
    {                                           \
      printf("cutensor: Error %s in line %d\n", \
             cutensorGetErrorString(err),       \
             __LINE__);                         \
      return false;                             \
    }                                           \
  }

#define CHECK_CALL(x, y)                                              \
  {                                                                   \
    auto status = (x);                                                \
    if (status != (y))                                                \
    {                                                                 \
      fprintf(stderr, "Check failed at %s:%d\n", __FILE__, __LINE__); \
      throw std::runtime_error("EINSUM_GEMM: Launch failed.");        \
    }                                                                 \
  }

template <typename U>
struct CuTensorTypeTraits;

template <>
struct CuTensorTypeTraits<double>
{
  static const cudaDataType_t cudaType = CUDA_R_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef double ScalarType;
};

template <>
struct CuTensorTypeTraits<float>
{
  static const cudaDataType_t cudaType = CUDA_R_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_TF32;
  typedef float ScalarType;
};

template <>
struct CuTensorTypeTraits<cuComplex>
{
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef cuComplex ScalarType;
};

template <>
struct CuTensorTypeTraits<cuDoubleComplex>
{
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef cuDoubleComplex ScalarType;
};

template <>
struct CuTensorTypeTraits<__half>
{
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

size_t getDataTypeSize(const cudaDataType_t type)
{
  if ((type == CUDA_R_8I) || (type == CUDA_R_8U))
  {
    return 1U;
  }
  else if (type == CUDA_R_16F)
  {
    return 2U;
  }
  else if ((type == CUDA_R_32I) || (type == CUDA_R_32U))
  {
    return 4U;
  }
  else if ((type == CUDA_R_32F) || (type == CUDA_C_16F))
  {
    return 4U;
  }
  else if ((type == CUDA_R_64F) || (type == CUDA_C_32F))
  {
    return 8U;
  }
  else if (type == CUDA_C_64F)
  {
    return 16U;
  }
  else
  {
    throw std::invalid_argument("Datatype is not yet supported.");
  }
}

struct GPUTimer
{
  GPUTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~GPUTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_, 0); }

  float seconds()
  {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }

private:
  cudaEvent_t start_, stop_;
};

template <typename ComputeType, typename IntType, int kMaxNumModes_>
struct Einsum
{
  static const std::vector<IntType> emptyVec;

  Einsum(const std::string &equation,
         const std::vector<IntType> &A_shape,
         const std::vector<IntType> &B_shape = emptyVec,
         const cutensorOperator_t opA = CUTENSOR_OP_IDENTITY,
         const cutensorOperator_t opB = CUTENSOR_OP_IDENTITY)
      : numModesA_(A_shape.size()),
        numModesB_(B_shape.size()),
        numModesC_(0),
        opA_(opA),
        opB_(opB),
        isInitialized_(false)
  {
    const auto arrow_pos = equation.find("->");
    const auto comma_pos = equation.find(",");
    const auto dots = equation.find("...");
    const bool isBroadcast = (dots != std::string::npos);
    const bool isImplicit = (arrow_pos == std::string::npos);
    if (isBroadcast)  // TODO
    {
      return;
    }
    const bool usesB = (comma_pos != std::string::npos);
    if (!usesB)
    {
      numModesB_ = 0;
    }

    size_t a_start = 0;
    size_t a_end =
        isImplicit
            ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos)
            : ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
    size_t b_start = usesB ? comma_pos + 1 : 0;
    size_t b_end = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
    size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
    size_t c_end = equation.size();

    char modeA[kMaxNumModes_ + 2];
    uint32_t numModesA = 0;
    for (int i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i)
    {
      if (equation.at(i) != ' ')  // skip spaces
      {
        modeA[numModesA++] = equation.at(i);
      }
    }

    char modeB[kMaxNumModes_ + 2];
    uint32_t numModesB = 0;
    for (int i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i)
    {
      if (equation.at(i) != ' ')  // skip spaces
      {
        modeB[numModesB++] = equation.at(i);
      }
    }

    char modeC[kMaxNumModes_ + 2];
    uint32_t numModesC = 0;
    for (int i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i)
    {
      if (equation.at(i) != ' ')  // skip spaces
      {
        modeC[numModesC++] = equation.at(i);
      }
    }

    if ((numModesA != numModesA_) || (numModesB != numModesB_))
    {
      // substring size and shape don't match
      return;
    }
    if (numModesA_ > kMaxNumModes_ || numModesB_ > kMaxNumModes_)
    {
      // too many modes
      return;
    }

    /**
     * Copy all modes from modeA to modeC if they don't appear in modeB
     */
    auto copyModesIf = [](const char *modeA,
                          uint32_t numModesA,
                          const char *modeB,
                          uint32_t numModesB,
                          char *modeC,
                          uint32_t &numModesC)
    {
      for (uint32_t i = 0; i < numModesA; i++)
      {
        auto mode = modeA[i];
        bool found = false;
        for (uint32_t j = 0; j < numModesB; ++j)
        {
          if (mode == modeB[j])
          {
            found = true;
            break;
          }
        }

        if (!found)  // is non-contracted mode
        {
          modeC[numModesC++] = mode;
          if (numModesC > kMaxNumModes_)
          {
            // too many modes
            return false;
          }
        }
      }
      return true;
    };

    std::array<char, kMaxNumModes_ + 1> implicitModeC;
    char *redirectModeC;
    if (isImplicit)
    {
      // we have to copy all non-contracted modes from A over to C
      if (copyModesIf(modeA,
                      numModesA_,
                      modeB,
                      numModesB_,
                      implicitModeC.data(),
                      numModesC_) == false)
      {
        return;
      }
      // we have to copy all non-contracted modes from B over to C
      if (copyModesIf(modeB,
                      numModesB_,
                      modeA,
                      numModesA_,
                      implicitModeC.data(),
                      numModesC_) == false)
      {
        return;
      }
      std::sort(
          implicitModeC.begin(),
          std::next(implicitModeC.begin(),
                    numModesC_));  // modes are sorted w.r.t. lexical order
      implicitModeC[numModesC_] = '\0';
      redirectModeC = implicitModeC.data();
    }
    else
    {
      redirectModeC = modeC;
      numModesC_ = numModesC;
    }

    for (uint32_t i = 0; i < numModesA_; i++)
    {
      modesA_[i] = modeA[numModesA_ - i - 1];
      extentA_[i] = A_shape[numModesA_ - i - 1];
    }

    for (uint32_t i = 0; i < numModesB_; i++)
    {
      modesB_[i] = modeB[numModesB_ - i - 1];
      extentB_[i] = B_shape[numModesB_ - i - 1];
    }

    for (uint32_t i = 0; i < numModesC_; i++)
    {
      const auto mode = redirectModeC[numModesC_ - i - 1];
      modesC_[i] = mode;
      bool found = false;
      for (uint32_t j = 0; j < numModesA_; ++j)
      {
        if (modesA_[j] == mode)
        {
          extentC_[i] = extentA_[j];
          found = true;
          break;
        }
      }
      for (uint32_t j = 0; !found && j < numModesB_; ++j)
      {
        if (modesB_[j] == mode)
        {
          extentC_[i] = extentB_[j];
          break;
        }
      }
    }

    isInitialized_ = true;
  }

  size_t getWorksize() const
  {
    size_t sizeC = 1;
    for (int i = 0; i < numModesC_; ++i) sizeC *= extentC_.at(i);

    size_t sizeA = 1;
    for (int i = 0; i < numModesA_; ++i) sizeA *= extentA_.at(i);

    size_t sizeB = 1;
    for (int i = 0; i < numModesB_; ++i) sizeB *= extentB_.at(i);

    const size_t dataTypeSize =
        getDataTypeSize(CuTensorTypeTraits<ComputeType>::cudaType);
    // return (sizeC + sizeA + sizeB) * dataTypeSize;
    return 1024ULL * 1024ULL * 1024ULL * 2ULL;
  }

  std::vector<IntType> getOutputShape() const
  {
    if (!isInitialized_) return {};
    std::vector<IntType> extentC(numModesC_);
    for (int i = 0; i < numModesC_; ++i)
    {
      extentC[i] = extentC_.at(numModesC_ - i - 1);
    }

    return extentC;
  }

  /**
   * Computes the einsum call A,B->C
   *
   * \param[in] A_raw device pointer of A
   * \param[in] B_raw device pointer of B
   * \param[out] C_raw device pointer of C
   * \param[out] wor_raw device pointer to the scratchpad memory
   * Dispatch to contraction
   */
  bool execute(const cutensorHandle_t *handle,
               void *opt_plan_raw,
               const void *A_raw,
               const void *B_raw,
               void *C_raw,
               void *work_raw,
               size_t worksize,
               cudaStream_t stream) const
  {
    // printf("here excute !!\n");
    // printf("opt_plan in excute is %d\n", *((int32_t *)opt_plan_raw));

    if (!isInitialized_) return false;

    cudaDataType_t cudaType = CuTensorTypeTraits<ComputeType>::cudaType;
    cutensorComputeType_t computeType =
        CuTensorTypeTraits<ComputeType>::cutensorType;

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                                              &descA,
                                              numModesA_,
                                              extentA_.data(),
                                              NULL /* = stride */,
                                              cudaType,
                                              opA_));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                                              &descC,
                                              numModesC_,
                                              extentC_.data(),
                                              NULL /* = stride*/,
                                              cudaType,
                                              CUTENSOR_OP_IDENTITY));

    uint32_t alignmentRequirementA;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(
        handle, A_raw, &descA, &alignmentRequirementA));

    uint32_t alignmentRequirementC;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(
        handle, C_raw, &descC, &alignmentRequirementC));

    cutensorTensorDescriptor_t descB;
    uint32_t alignmentRequirementB;
    if (numModesB_ > 0)
    {
      // dispatch to contraction
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                                                &descB,
                                                numModesB_,
                                                extentB_.data(),
                                                NULL /* = stride*/,
                                                cudaType,
                                                opB_));

      HANDLE_ERROR(cutensorGetAlignmentRequirement(
          handle, B_raw, &descB, &alignmentRequirementB));

      cutensorContractionDescriptor_t desc;
      HANDLE_ERROR(cutensorInitContractionDescriptor(handle,
                                                     &desc,
                                                     &descA,
                                                     modesA_.data(),
                                                     alignmentRequirementA,
                                                     &descB,
                                                     modesB_.data(),
                                                     alignmentRequirementB,
                                                     &descC,
                                                     modesC_.data(),
                                                     alignmentRequirementC,
                                                     &descC,
                                                     modesC_.data(),
                                                     alignmentRequirementC,
                                                     computeType));

      // cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
      cutensorContractionFind_t find;
      HANDLE_ERROR(
          cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

#if 0
      uint64_t worksize = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspace(
          handle, &desc, &find, CUTENSOR_WORKSPACE_MAX, &worksize));

      // printf("worksize is %lld\n", (long long)worksize);

      void *work = nullptr;
      if (worksize > 0)
      {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
          // try with recommended size
          HANDLE_ERROR(cutensorContractionGetWorkspace(
              handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
          if (cudaSuccess != cudaMalloc(&work, worksize))
          {
            printf("workspace allocation fails with size %lld\n", worksize);
            uint64_t free, total;
            cudaMemGetInfo(&free, &total);
            free /= (1024 * 1024 * 1024);
            total /= (1024 * 1024 * 1024);
            printf("free size %lld gb, total size %lld gb\n", free, total);
            work = nullptr;
            worksize = 0;
            return false;
          }
        }
      }
#else
      // uint64_t worksize = kWorksize_;
      void *work = work_raw;
#endif

      // at::Tensor workspace = at::empty({static_cast<int>(worksize)},
      // at::CUDA(at::kByte)); void *work = workspace.data_ptr<uint8_t>();

      // printf("kWorksize_ is %lld\n", (long long)kWorksize_);

      cutensorContractionPlan_t plan;
      HANDLE_ERROR(
          cutensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

      typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
      typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;

      int32_t maxAlgosTC = 0;
      cutensorContractionMaxAlgos(&maxAlgosTC);
      // printf("maximum avaliable algorithms is %d\n", maxAlgosTC);

      // printf("Contraction count: %d\n", contraction_count);
      if (*((int32_t *)opt_plan_raw) == -99)
      {
        printf("Contraction count: %d\n", contraction_count);
        double bestTime = 1e100;
        int bestAlgo = -1;
        double deaultTimeCUTENSOR = 1e100;

        // compute transferred bytes
        uint64_t elementsA = 1;
        for (uint32_t i = 0; i < numModesA_; i++)
        {
          elementsA *= extentA_[i];
        }
        uint64_t elementsB = 1;
        for (uint32_t i = 0; i < numModesB_; i++)
        {
          elementsB *= extentB_[i];
        }
        uint64_t elementsC = 1;
        for (uint32_t i = 0; i < numModesC_; i++)
        {
          elementsC *= extentC_[i];
        }

        double sizeall = sizeof(ComputeType) *
                         (elementsA + elementsB + elementsC) /
                         (1024. * 1024. * 1024.);
        printf("Total memory: %.2f MB\n", sizeall * 1024);
        printf("cuTensor workspace: %.2f MB\n", worksize / (1024. * 1024.));

        // count gflops
        double gflops = 6.0;
        for (uint32_t i = 0; i < numModesA_; i++)
        {
          gflops *= extentA_[i];
        }

        for (uint32_t i = 0; i < numModesB_; i++)
        {
          bool find = false;
          for (uint32_t j = 0; j < numModesA_; j++)
          {
            if (modesB_[i] == modesA_[j])
            {
              find = true;
              break;
            }
          }
          if (!find) gflops *= extentB_[i];
        }

        gflops /= 1.e9;

        // for (int algo = (int)CUTENSOR_ALGO_GETT; algo < maxAlgosTC; algo++)
        for (int algo = (int)CUTENSOR_ALGO_GETT; algo < maxAlgosTC; algo++)
        {
          double minTimeCUTENSOR = 1e100;
          cutensorStatus_t err;
          for (int i = 0; i < 3; i++)
          {
            err = cutensorInitContractionFind(
                handle, &find, (cutensorAlgo_t)algo);

            if (err == CUTENSOR_STATUS_SUCCESS)
            {
              err = cutensorInitContractionPlan(
                  handle, &plan, &desc, &find, worksize);

              if (err == CUTENSOR_STATUS_SUCCESS)
              {
                // Set up timing
                GPUTimer timer;
                timer.start();

                err = cutensorContraction(handle,
                                          &plan,
                                          (void *)&alpha,
                                          A_raw,
                                          B_raw,
                                          (void *)&beta,
                                          C_raw,
                                          C_raw,
                                          work,
                                          worksize,
                                          stream);

                // Synchronize and measure timing
                auto time = timer.seconds();

                minTimeCUTENSOR =
                    (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
              }
            }
          }

          if (err == CUTENSOR_STATUS_SUCCESS)
          {
            printf(
                "cuTensor: algo %3d is %4.2f ms and %.2f GB/s and %.2f "
                "GFLOPs/s\n",
                algo,
                minTimeCUTENSOR * 1000,
                sizeall / minTimeCUTENSOR,
                gflops / minTimeCUTENSOR);

            if ((cutensorAlgo_t)algo == CUTENSOR_ALGO_DEFAULT)
              deaultTimeCUTENSOR = minTimeCUTENSOR;
          }

          if (bestTime > minTimeCUTENSOR)
          {
            bestTime = minTimeCUTENSOR;
            bestAlgo = algo;
          }
        }
        /*************************/

        printf(
            "best:     algo %3d is %4.2f ms and %.2f GB/s and %.2f GFLOPs/s\n",
            bestAlgo,
            bestTime * 1000,
            sizeall / bestTime,
            gflops / bestTime);
        printf("saved:    %4.4f ms\n\n",
               (deaultTimeCUTENSOR - bestTime) * 1000);
        *((int32_t *)opt_plan_raw) = bestAlgo;
      }
      else
      {
        // printf("following best algo is %d\n", *((int32_t *)opt_plan_raw));
        cutensorContractionFind_t find;
        HANDLE_ERROR(cutensorInitContractionFind(
            handle, &find, (cutensorAlgo_t)(*((int32_t *)opt_plan_raw))));

        cutensorContractionPlan_t plan;
        HANDLE_ERROR(
            cutensorInitContractionPlan(handle, &plan, &desc, &find, worksize));

        // float time_estimated = 0.0f;
        // HANDLE_ERROR(cutensorContractionEstimateRuntime(
        //     handle, &desc, CUTENSOR_ALGO_DEFAULT, &time_estimated));
        // printf("cuTensor: estimated time = %.6f ms\n", time_estimated);

        // GPUTimer timer;
        // timer.start();
        HANDLE_ERROR(cutensorContraction(handle,
                                         &plan,
                                         (void *)&alpha,
                                         A_raw,
                                         B_raw,
                                         (void *)&beta,
                                         C_raw,
                                         C_raw,
                                         work,
                                         worksize,
                                         stream));
        // auto time = timer.seconds();
        // printf("cuTensor: %d algo %.6f ms\n",
        //        *((int32_t *)opt_plan_raw),
        //        time * 1000);

        // HANDLE_ERROR(cutensorContraction(handle,
        //                                  &plan,
        //                                  (void *)&alpha,
        //                                  A_raw,
        //                                  B_raw,
        //                                  (void *)&beta,
        //                                  C_raw,
        //                                  C_raw,
        //                                  work_raw,
        //                                  kWorksize_,
        //                                  stream));
      }
      contraction_count++;

      // if (work) cudaFree(work);
    }
    else
    {
      // dispatch to reduction
      typename CuTensorTypeTraits<ComputeType>::ScalarType alpha = 1;
      typename CuTensorTypeTraits<ComputeType>::ScalarType beta = 0;
      HANDLE_ERROR(
          cutensorReduction(handle,
                            (const void *)&alpha,
                            A_raw,
                            &descA,
                            modesA_.data(),
                            (const void *)&beta,
                            A_raw,
                            &descC,
                            modesC_.data(),  // beta == 0 => will not be used
                            C_raw,
                            &descC,
                            modesC_.data(),
                            CUTENSOR_OP_ADD,
                            computeType,
                            work_raw,
                            worksize,
                            stream));
    }
    return true;
  }

  bool isInitialized() const { return isInitialized_; }

private:
  // static const size_t kWorksize_ = 1024ULL * 1024ULL * 8ULL * 128ULL;
  // static const size_t kWorksize_ = 20096679424ULL;
  uint32_t numModesA_;
  uint32_t numModesB_;
  uint32_t numModesC_;
  bool isInitialized_;
  std::array<int, kMaxNumModes_> modesA_;
  std::array<int, kMaxNumModes_> modesB_;
  std::array<int, kMaxNumModes_> modesC_;
  std::array<int64_t, kMaxNumModes_> extentA_;
  std::array<int64_t, kMaxNumModes_> extentB_;
  std::array<int64_t, kMaxNumModes_> extentC_;
  cutensorOperator_t opA_ = CUTENSOR_OP_IDENTITY;
  cutensorOperator_t opB_ = CUTENSOR_OP_IDENTITY;
};

#if use_different_handles
inline cutensorHandle_t CreateCuTensorHandle_fp32()
{
  auto v_tf32 = getenv("NVIDIA_TF32_OVERRIDE");
  auto v_3xtf32 = getenv("CUTENSOR_ENABLE_3XTF32");
  setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
  setenv("CUTENSOR_ENABLE_3XTF32", "0", 1);
  cutensorHandle_t handle;
  CHECK_CALL(cutensorInit(&handle), CUTENSOR_STATUS_SUCCESS);
  if (getenv("CUTENSOR_CACHE") && atoi(getenv("CUTENSOR_CACHE")) == 1)
  {
    cutensorPlanCacheline_t *cachelines = new cutensorPlanCacheline_t[32];
    cutensorHandleAttachPlanCachelines(&handle, cachelines, 32);
  }
  setenv("NVIDIA_TF32_OVERRIDE", v_tf32, 1);
  setenv("CUTENSOR_ENABLE_3XTF32", v_3xtf32, 1);
  return handle;
}

inline cutensorHandle_t CreateCuTensorHandle_tf32()
{
  auto v_tf32 = getenv("NVIDIA_TF32_OVERRIDE");
  auto v_3xtf32 = getenv("CUTENSOR_ENABLE_3XTF32");
  setenv("NVIDIA_TF32_OVERRIDE", "1", 1);
  setenv("CUTENSOR_ENABLE_3XTF32", "0", 1);
  cutensorHandle_t handle;
  CHECK_CALL(cutensorInit(&handle), CUTENSOR_STATUS_SUCCESS);
  if (getenv("CUTENSOR_CACHE") && atoi(getenv("CUTENSOR_CACHE")) == 1)
  {
    cutensorPlanCacheline_t *cachelines = new cutensorPlanCacheline_t[32];
    cutensorHandleAttachPlanCachelines(&handle, cachelines, 32);
  }
  setenv("NVIDIA_TF32_OVERRIDE", v_tf32, 1);
  setenv("CUTENSOR_ENABLE_3XTF32", v_3xtf32, 1);
  return handle;
}

inline cutensorHandle_t CreateCuTensorHandle_3xtf32()
{
  auto v_tf32 = getenv("NVIDIA_TF32_OVERRIDE");
  auto v_3xtf32 = getenv("CUTENSOR_ENABLE_3XTF32");
  setenv("NVIDIA_TF32_OVERRIDE", "1", 1);
  setenv("CUTENSOR_ENABLE_3XTF32", "1", 1);
  cutensorHandle_t handle;
  CHECK_CALL(cutensorInit(&handle), CUTENSOR_STATUS_SUCCESS);
  if (getenv("CUTENSOR_CACHE") && atoi(getenv("CUTENSOR_CACHE")) == 1)
  {
    cutensorPlanCacheline_t *cachelines = new cutensorPlanCacheline_t[32];
    cutensorHandleAttachPlanCachelines(&handle, cachelines, 32);
  }
  setenv("NVIDIA_TF32_OVERRIDE", v_tf32, 1);
  setenv("CUTENSOR_ENABLE_3XTF32", v_3xtf32, 1);
  return handle;
}

static bool cutensor_init_fp32 = false;
static bool cutensor_init_tf32 = false;
static bool cutensor_init_3xtf32 = false;
static cutensorHandle_t handle_fp32;
static cutensorHandle_t handle_tf32;
static cutensorHandle_t handle_3xtf32;

inline cutensorHandle_t *GetCuTensorHandle()
{
  if (getenv("NVIDIA_TF32_OVERRIDE") &&
      atoi(getenv("NVIDIA_TF32_OVERRIDE")) == 0)
  {
    // printf("0 0\n");
    if (!cutensor_init_fp32)
    {
      handle_fp32 = CreateCuTensorHandle_fp32();
      cutensor_init_fp32 = true;
    }
    return &handle_fp32;
  }

  if (getenv("NVIDIA_TF32_OVERRIDE") &&
      atoi(getenv("NVIDIA_TF32_OVERRIDE")) == 1 &&
      getenv("CUTENSOR_ENABLE_3XTF32") &&
      atoi(getenv("CUTENSOR_ENABLE_3XTF32")) == 0)
  {
    // printf("1 0\n");
    if (!cutensor_init_tf32)
    {
      handle_tf32 = CreateCuTensorHandle_tf32();
      cutensor_init_tf32 = true;
    }
    return &handle_tf32;
  }

  if (getenv("NVIDIA_TF32_OVERRIDE") &&
      atoi(getenv("NVIDIA_TF32_OVERRIDE")) == 1 &&
      getenv("CUTENSOR_ENABLE_3XTF32") &&
      atoi(getenv("CUTENSOR_ENABLE_3XTF32")) == 1)
  {
    // printf("1 1\n");
    if (!cutensor_init_3xtf32)
    {
      handle_3xtf32 = CreateCuTensorHandle_3xtf32();
      cutensor_init_3xtf32 = true;
    }
    return &handle_3xtf32;
  }
}
#else
inline cutensorHandle_t CreateCuTensorHandle()
{
  cutensorHandle_t handle;
  cutensorInit(&handle);
  if (getenv("CUTENSOR_CACHE") && atoi(getenv("CUTENSOR_CACHE")) == 1)
  {
    cutensorPlanCacheline_t *cachelines = new cutensorPlanCacheline_t[32];
    cutensorHandleAttachPlanCachelines(&handle, cachelines, 32);
  }
  return handle;
}

inline cutensorHandle_t *GetCuTensorHandle()
{
  static cutensorHandle_t handle = CreateCuTensorHandle();
  return &handle;
}
#endif

static bool cublas_init = false;
static cublasHandle_t handle_cublas;
inline cublasHandle_t *GetCuBlasHandle()
{
  if (!cublas_init)
  {
    CHECK_CALL(cublasCreate(&handle_cublas), CUBLAS_STATUS_SUCCESS);
    cublas_init = true;
  }
  return &handle_cublas;
}

void einsum_gemm_execute(const int m,
                         const int n,
                         const int k,
                         const void *A,
                         const void *B,
                         void *C,
                         const int batch_count = 1,
                         const bool useCutlass = false);