/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Extension Libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>

//
// CUTLASS includes needed for single-precision GEMM kernel
//
#include "/root/cutlass/cutlass/cutlass.h"
// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "/root/cutlass/cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "/root/cutlass/cutlass/gemm/sgemm_traits.h"

#pragma warning( disable : 4503)

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size.
  //
  // Note, GemmTraits<> is a generic template defined for various general matrix product
  // computations within CUTLASS. It is intended to be maximally flexible, and consequently
  // it contains numerous template arguments.
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
  //
  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
    cutlass::Shape<8, 128, 128>            // threadblock tile size
  >
    GemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  //
  // One of CUTLASS's design patterns is to define parameters objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  typename Gemm::Params params;

  int result = params.initialize(
    M,     // GEMM M dimension
    N,     // GEMM N dimension
    K,     // GEMM K dimension
    alpha, // scalar alpha
    A,     // matrix A operand
    lda,
    B,     // matrix B operand
    ldb,
    beta,  // scalar beta
    C,     // source matrix C
    ldc,
    C,     // destination matrix C (may be different memory than source C matrix)
    ldc
  );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * ldm * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  //result = cudaMemset(*matrix, 0, sizeof_matrix);

  //if (result != cudaSuccess) {
  //  std::cerr << "Failed to clear matrix device memory: "
  //    << cudaGetErrorString(result) << std::endl;
  //  return result;
  //}

  // Initialize matrix elements to arbitrary small integers.
  //result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  //if (result != cudaSuccess) {
  //  std::cerr << "Failed to initialize matrix: "
  //    << cudaGetErrorString(result) << std::endl;
  //  return result;
  //}

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta,
			    torch::Tensor W,		
			    torch::Tensor I,		
			    torch::Tensor& O) {
  cudaError_t result = cudaSuccess;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  //size_t sizeof_A = sizeof(float) * lda * K;
  //size_t sizeof_B = sizeof(float) * ldb * N;
  //size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  //float *A;
  //float *B;
  //float *C;
  //float *C_reference;

  torch::Device deviceCPU(torch::kCPU);
  torch::Device deviceGPU(torch::kCPU);
  //torch::Device deviceGPU;

  if (torch::cuda::is_available())
  {
      std::cout << "CUDA is available! Run on GPU." << std::endl;
      deviceGPU = torch::Device(torch::kCUDA);

  }
  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  // Note: initialization is done to 0. Removed seed initilization.

  //result = AllocateMatrix(&A, lda, M, K, 0);
  //std::cout << W << std::endl;

  // Allocate device memory.
  torch::Tensor A = W.to(deviceGPU,at::kFloat,false,true); // Tensor.to(deviceCPU/GPU, ScalarType, non_blocking, copy)
  //result = cudaMemcpy(A, static_cast<float*>(W.data_ptr()), sizeof_A, cudaMemcpyHostToDevice);

  if (result !=  cudaSuccess) {
    std::cerr << "Failed to copy Weight matrix to A: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }
  else std::cout << "Weight Matrix copied to A successfully." << std::endl;

  //result = AllocateMatrix(&B, ldb, K, N, 17);
  torch::Tensor B = I.to(deviceGPU,at::kFloat,false,true); // Tensor.to(deviceCPU/GPU, ScalarType, non_blocking, copy)
  //result = cudaMemcpy(B, static_cast<const float*>(I.data_ptr()), sizeof_B, cudaMemcpyHostToDevice);

  if (result !=  cudaSuccess) {
    std::cerr << "Failed to copy Input matrix to B: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(A.data_ptr());
    return result;
  }
  else std::cout << "Input Matrix copied to B successfully." << std::endl;

  //result = cudaMemcpy(C, static_cast<float*>(O.data_ptr()), sizeof_C, cudaMemcpyHostToDevice);
  torch::Tensor C = O.to(deviceGPU,at::kFloat,false,true); // Tensor.to(deviceCPU/GPU, ScalarType, non_blocking, copy)
  //result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Output matrix to C: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(A.data_ptr());
    cudaFree(B.data_ptr());
    return result;
  }
  else std::cout << "(Initial) Output Matrix copied to C successfully." << std::endl;

  //result = AllocateMatrix(&C_reference, ldc, M, N, 101);

  //if (result != cudaSuccess) {
  //  cudaFree(A);
  //  cudaFree(B);
  //  cudaFree(C_cutlass);
  //  return result;
  //}

  //result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  //if (result != cudaSuccess) {
  //  std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
  //    << cudaGetErrorString(result) << std::endl;

  //  cudaFree(C_reference);
  //  cudaFree(C_cutlass);
  //  cudaFree(B);
  //  cudaFree(A);

  //  return result;
  //}

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, static_cast<const float*>(A.data_ptr()), lda, static_cast<const float*>(B.data_ptr()), ldb, beta, static_cast<float*>(C.data_ptr()), ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    //cudaFree(C_reference);
    cudaFree(C.data_ptr());
    cudaFree(B.data_ptr());
    cudaFree(A.data_ptr());

    return result;
  }
  else std::cout << "GEMM launched successfully." << std::endl;

  //
  // Verify.
  //

  // Launch reference GEMM
  //result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  //if (result != cudaSuccess) {
  //  std::cerr << "Reference GEMM kernel failed: "
  //    << cudaGetErrorString(result) << std::endl;

  //  cudaFree(C_reference);
  //  cudaFree(C_cutlass);
  //  cudaFree(B);
  //  cudaFree(A);

  //  return result;
  //}

  // Copy to host and verify equivalence.
  //std::vector<float> host_cutlass(ldc * N, 0);
  //std::vector<float> host_reference(ldc * N, 0);
  //result = cudaMemcpy(host_cutlass.data(), C, sizeof_C, cudaMemcpyDeviceToHost);
  //result = cudaMemcpy(static_cast<float*>(O.data_ptr()), C, sizeof_C, cudaMemcpyDeviceToHost);
  std::cout << "C\n";
  std::cout << C << std::endl;
  std::cout << "O before\n";
  std::cout << O << std::endl;
  O.copy_(C);//.to(deviceCPU,at::kFloat,false,true); // Tensor.to(deviceCPU/GPU, ScalarType, non_blocking, copy)
  std::cout << "O after\n";
  std::cout << O << std::endl;
  // TODO: How to pass the data to the CPP? What to do with the result?
  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    //cudaFree(C_reference);
    cudaFree(C.data_ptr());
    cudaFree(B.data_ptr());
    cudaFree(A.data_ptr());

    return result;
  }
  else std::cout << "(Final) C Matrix copied to Output Matrix successfully." << std::endl;

  //result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  //if (result != cudaSuccess) {
  //  std::cerr << "Failed to copy Reference GEMM results: "
  //    << cudaGetErrorString(result) << std::endl;

  //  cudaFree(C_reference);
  //  cudaFree(C_cutlass);
  //  cudaFree(B);
  //  cudaFree(A);

  //  return result;
  //}

  //
  // Free device memory allocations.
  //

  //cudaFree(C.data_ptr());
  //cudaFree(B.data_ptr());
  //cudaFree(A.data_ptr());
  //
  // Test for bit equivalence of results.
  //

  //if (host_cutlass != host_reference) {
  //  std::cerr << "CUTLASS results incorrect." << std::endl;

  //  return cudaErrorUnknown;
  //}

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int basic_gemm_cuda(torch::Tensor W, torch::Tensor I, torch::Tensor& O) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3];
  problem[0] = static_cast<int>(W.sizes()[1]); // M - out_ch
  problem[1] = static_cast<int>(I.sizes()[0]); // N - t*b
  problem[2] = static_cast<int>(W.sizes()[0]); // K - in_ch

  //for (int i = 1; i < argc && i < 4; ++i) {
  //  std::stringstream ss(arg[i]);
  //  ss >> problem[i - 1];
  //}

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 1 };

  //for (int i = 4; i < argc && i < 6; ++i) {
  //  std::stringstream ss(arg[i]);
  //  ss >> scalars[i - 4];
  //}
  
  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],     // beta
    W,              // weight tensor
    I,              // Input tensor
    O	            // Output tensor
    );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
