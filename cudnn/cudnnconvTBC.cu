// Let the GPU perform a 2-dimension convolution over filter KIO on tensor TBC
// To compile: nvcc -arch=sm_70 -std=c++14 -o conv_tbc cudnnconvTBC.cu -lcudnn
// Example command: ./conv_tbc testW testI
// Last modified: Bambo Wu 05/09/2019
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cudnn.h>

std::string CONV_FWD_ALGO[8] = {
  "IMPLICIT_GEMM",
  "IMPLICIT_PRECOMP_GEMM",
  "GEMM",
  "DIRECT",
  "FFT",
  "FFT_TILING",
  "WINOGRAD",
  "WINOGRAD_NONFUSED"};

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

int offsetTBC(int *dims, int *coo) {
  //int T = dims[0];
  int B = dims[1];
  int C = dims[2];
  int t = coo[0];
  int b = coo[1];
  int c = coo[2];
  return c + b * C + t * B * C;
}

int offsetKIO(int *dims, int *coo) {
  //int K = dims[0];
  int I = dims[1];
  int O = dims[2];
  int k = coo[0];
  int i = coo[1];
  int o = coo[2];
  return o + i * O + k * I * O;
}

// reorganize weights from KIO to OIK, Win, Wout should no overlap
void reorg_weights(int *Wdims, float *Win, float *Wout) {
  int offset = 0;
  for (int k = 0; k < Wdims[0]; ++k) {
    for (int in_ch = 0; in_ch < Wdims[1]; ++in_ch) {
      for (int out_ch = 0; out_ch < Wdims[2]; ++out_ch) {
        int offset_new = k + in_ch * Wdims[0] + out_ch * Wdims[0] * Wdims[1];
        Wout[offset_new] = Win[offset++];
      }
    }
  }
}

//// reorganize weights from KIO to OKI, Win, Wout should no overlap
//void reorg_weights(int *Wdims, float *Win, float *Wout) {
//  int offset = 0;
//  for (int k = 0; k < Wdims[0]; ++k) {
//    for (int in_ch = 0; in_ch < Wdims[1]; ++in_ch) {
//      for (int out_ch = 0; out_ch < Wdims[2]; ++out_ch) {
//        int offset_new = in_ch + k * Wdims[1] + out_ch * Wdims[0] * Wdims[1];
//        Wout[offset_new] = Win[offset++];
//      }
//    }
//  }
//}

int main(int argc, char *argv[]) {

  int pads = 0;
  if (3 > argc) {
    std::cout << "Usage: " << argv[0] << " <W> <I> [pads=0]\n";
    return -1;
  } else if (3 < argc) {
    pads = atoi(argv[3]);
  }

  //std::cout << cudnnGetVersion() << std::endl;

  // read the filter from file
  std::string dummy_line;
  int Wdims[3]; // {K, I, O}
  std::ifstream ifs(argv[1]);
  getline(ifs, dummy_line); // dummy line with "K   I   O"
  ifs >> Wdims[0] >> Wdims[1] >> Wdims[2];
  int Wsize = Wdims[0] * Wdims[1] * Wdims[2];
  float *Wdata = new float[Wsize];
  for (int out_ch = 0; out_ch < Wdims[2]; ++out_ch) {
    for (int in_ch = 0; in_ch < Wdims[1]; ++in_ch) {
      for (int k = 0; k < Wdims[0]; ++k) {
        int Wcoo[3] = {k, in_ch, out_ch};
        int offset = offsetKIO(Wdims, Wcoo);
        ifs >> Wdata[offset];
      }
    }
  }
  ifs.close();

  for (int out_ch = 0; out_ch < Wdims[2]; ++out_ch) {
    std::cout << "\033[1mWeights\033[3" << (6 - out_ch) << "m[out_ch=" <<
      out_ch << "]\033[0m\n";
    for (int in_ch = 0; in_ch < Wdims[1]; ++in_ch) {
      std::cout << "\033[9" << in_ch + 1 << "m[in_ch=" <<
        in_ch << "]\033[0m[0:" << Wdims[0] - 1 << "]: ";
      for (int k = 0; k < Wdims[0]; ++k) {
        int Wcoo[3] = {k, in_ch, out_ch};
        int offset = offsetKIO(Wdims, Wcoo);
        std::cout << std::setw(7) << std::fixed <<
          std::setprecision(4) << Wdata[offset] << "  ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // read the filter from file
  int Idims[3]; // {T, B, C}
  ifs.open(argv[2]);
  getline(ifs, dummy_line); // dummy line with "T   B   C"
  ifs >> Idims[0] >> Idims[1] >> Idims[2];
  int Isize = Idims[0] * Idims[1] * Idims[2];
  float *Idata = new float[Isize];
  for (int batch = 0; batch < Idims[1]; ++batch) {
    for (int in_ch = 0; in_ch < Idims[2]; ++in_ch) {
      for (int t = 0; t < Idims[0]; ++t) {
        int Icoo[3] = {t, batch, in_ch};
        int offset = offsetTBC(Idims, Icoo);
        ifs >> Idata[offset];
      }
    }
  }
  ifs.close();

  for (int batch = 0; batch < Idims[1]; ++batch) {
    std::cout << "\033[1mInputs[batch=" << batch << "]\033[0m\n";
    for (int in_ch = 0; in_ch < Idims[2]; ++in_ch) {
      std::cout << "\033[9" << in_ch + 1 << "m[in_ch=" <<
        in_ch << "]\033[0m[0:" << Idims[0] - 1 << "]: ";
      for (int t = 0; t < Idims[0]; ++t) {
        int Icoo[3] = {t, batch, in_ch};
        int offset = offsetTBC(Idims, Icoo);
        std::cout << std::setw(7) << std::fixed <<
          std::setprecision(4) << Idata[offset] << "  ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // reorganize weights and inputs
  float *Wdata4D = new float[Wsize];
  reorg_weights(Wdims, Wdata, Wdata4D);
  int Wdims4D[4] = {Wdims[2], // output channels
                    Wdims[1], // input channels
                    1, // to makeup 4D filter for convolution
                    Wdims[0]}; // filter length K
  int Idims4D[4] = {Idims[1], // batch
                    Idims[2], // input channels
                    1, // to makeup 4D tensor for convolution
                    Idims[0]}; // time
  int Istride[4] = {Idims[2], // next bach comes after all in_ch
                    1, // next channels is stored right after
                    1, // will not get non-zero coo on this dimension
                    Idims[1] * Idims[2]}; // next time point is after B*C


  int Odims4D[4] = {Idims[1], // B of outputs equals to B of inputs
                    Wdims[2], // C of outputs equals to O of weights
                    1, // to makeup 4D tensor for convolution
                    Idims[0] - Wdims[0] + 1 + 2 * pads}; // T of outputs
  int Ostride[4] = {Odims4D[1], // next batch comes after all in_ch
                    1, // next channel is stored to the next
                    1, // will not got non-zero coo on this dimension
                    Odims4D[0] * Odims4D[1]}; // next time point is after B*C
  int Osize = Odims4D[0] * Odims4D[1] * Odims4D[3];
  float *Odata = new float[Osize];

  int pad0s[2] = {0, pads};
  int dilation1s[2] = {1, 1}; // dilations
  int stride1s[2] = {1, 1}; // strides

  // create CuDNN context
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  // create convolution layer
  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolutionNdDescriptor(conv_desc,         // handle
                                             2,                 // arrayLength
                                             pad0s,             // paddings
                                             stride1s,          // stride
                                             dilation1s,        // dilation
                                             CUDNN_CROSS_CORRELATION ,// mode
                                             CUDNN_DATA_FLOAT));//precision
  // create filter
  cudnnFilterDescriptor_t fil_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&fil_desc));
  checkCUDNN(cudnnSetFilterNdDescriptor(fil_desc,               // descriptor
                                        CUDNN_DATA_FLOAT,       // precision
                                        CUDNN_TENSOR_NCHW,      // layout
                                        4,                      // dimension
                                        Wdims4D));              // filter size
  // create the tensor for input
  cudnnTensorDescriptor_t in_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
  checkCUDNN(cudnnSetTensorNdDescriptor(in_desc,                // descriptor
                                        CUDNN_DATA_FLOAT,       // precision
                                        4,                      // dimension
                                        Idims4D,                // size
                                        Istride));              // stride
  // create the tensor for output
  cudnnTensorDescriptor_t out_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
  checkCUDNN(cudnnSetTensorNdDescriptor(out_desc,               // descriptor
                                        CUDNN_DATA_FLOAT,       // precision
                                        4,                      // dimension
                                        Odims4D,                // size
                                        Ostride));              // stride
  //// find convolution algorithm
  int algos_cnt = 8;
  cudnnConvolutionFwdAlgoPerf_t conv_algos[8];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                                  in_desc,
                                                  fil_desc,
                                                  conv_desc,
                                                  out_desc,
                                                  algos_cnt,
                                                  &algos_cnt,
                                                  conv_algos));
  bool isfeasible = false;
  for (int i = 0; i < algos_cnt && 0 == conv_algos[i].status; ++i) {
    std::cout << "cudnnConvolutionFwdAlgo_t: " <<
      CONV_FWD_ALGO[conv_algos[i].algo] <<
      " (" << conv_algos[i].time << " ms)\n";
    std::cout << "workspace: " << conv_algos[i].memory << " "
      "determinism: " << conv_algos[i].determinism << " "
      "mathType: " << conv_algos[i].mathType << std::endl;
    isfeasible = true;
  }
  std::cout << std::endl;

  if (!isfeasible) {
    // dimensions check
    std::cout << "Layout:  {N, H, W, C}\n";
    std::cout << "Wdims4D: {" << Wdims4D[0] << ", " << Wdims4D[1] <<
      ", " << Wdims4D[2] << ", " << Wdims4D[3] << "}\n";

    std::cout << "Idims4D: {" << Idims4D[0] << ", " << Idims4D[1] <<
      ", " << Idims4D[2] << ", " << Idims4D[3] << "}\n";

    std::cout << "Odims4D: {" << Odims4D[0] << ", " << Odims4D[1] <<
      ", " << Odims4D[2] << ", " << Odims4D[3] << "}\n";

    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(conv_desc,
                                                     in_desc,
                                                     fil_desc,
                                                     4,
                                                     Odims4D));

    std::cout << "Supposed to be\n";
    std::cout << "Odims4D: {" << Odims4D[0] << ", " << Odims4D[1] <<
      ", " << Odims4D[2] << ", " << Odims4D[3] << "}\n";
  } else {
    // allocate memory on GPU
    float *Wdata_GPU;
    float *Idata_GPU;
    float *Odata_GPU;
    char *workspace = NULL;
    cudaMalloc(&Wdata_GPU, Wsize * sizeof(float));
    cudaMalloc(&Idata_GPU, Isize * sizeof(float));
    cudaMalloc(&Odata_GPU, Osize * sizeof(float));
    if (conv_algos[0].memory) {
      cudaMalloc(&workspace, conv_algos[0].memory);
    }

    // prepare data on GPU and compute
    float alpha = 1.0;
    float beta = 1.0;
    cudaMemcpy(Wdata_GPU, Wdata4D, Wsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Idata_GPU, Idata, Isize * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       in_desc,
                                       Idata_GPU,
                                       fil_desc,
                                       Wdata_GPU,
                                       conv_desc,
                                       conv_algos[0].algo,
                                       &workspace,
                                       conv_algos[0].memory,
                                       &beta,
                                       out_desc,
                                       Odata_GPU));
    cudaMemcpy(Odata, Odata_GPU, Osize * sizeof(float), cudaMemcpyDeviceToHost);

    if (conv_algos[0].memory) {
      cudaFree(workspace);
    }
    cudaFree(Odata_GPU);
    cudaFree(Idata_GPU);
    cudaFree(Wdata_GPU);

    //int Odims[3] = {Odims4D[2], Odims4D[0], Odims4D[3]};
    int Odims[3] = {Odims4D[3], Odims4D[0], Odims4D[1]};
    for (int batch = 0; batch < Odims[1]; ++batch) {
      std::cout << "\033[1mOutputs[batch=" << batch << "]\033[0m\n";
      for (int out_ch = 0; out_ch < Odims[2]; ++out_ch) {
        std::cout << "\033[3" << (6 - out_ch) << "m[out_ch=" <<
          out_ch << "]\033[0m[0:" << Odims[0] - 1 << "]: ";
        for (int t = 0; t < Odims[0]; ++t) {
          int Ocoo[3] = {t, batch, out_ch};
          int offset = offsetTBC(Odims, Ocoo);
          std::cout << std::setw(7) << std::fixed <<
            std::setprecision(4) << Odata[offset] << "  ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  delete[] Wdata, Idata, Odata, Wdata4D;
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyFilterDescriptor(fil_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn);

  return 0;

}
