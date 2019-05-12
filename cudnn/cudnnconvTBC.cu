// Let the GPU perform a 2-dimension convolution over filter KIO on tensor TBC
// To compile: nvcc -arch=sm_70 -std=c++14 -o conv_tbc cudnnconvTBC.cu -lcudnn
// Example command: ./conv_tbc testW testB testI testL
// Last modified: Bambo Wu 05/12/2019
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

std::string FILTER_BWD_ALGO[6] = {
  "BWD_FILTER_ALGO_0",
  "BWD_FILTER_ALGO_1",
  "FFT",
  "BWD_FILTER_ALGO_3",
  "WINOGRAD_NONFUSED",
  "FFT_TILING"};

std::string DATA_BWD_ALGO[6] = {
  "BWD_DATA_ALGO_0",
  "BWD_DATA_ALGO_1",
  "FFT",
  "FFT_TILING",
  "WINOGRAD",
  "WINOGRAD_NONFUSED"};

//function to print out error message from CUDA calls
#define checkCUDA(exp) \
  { \
    cudaError_t status = (exp); \
    if(status != cudaSuccess) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  }

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

// reorganize loss from TBC to NCHW, Lin, Lout should no overlap
void reorg_loss(int *Ldims, float *Lin, float *Lout) {
  int offset = 0;
  for (int t = 0; t < Ldims[0]; ++t) {
    for (int batch = 0; batch < Ldims[1]; ++batch) {
      for (int ch = 0; ch < Ldims[2]; ++ch) {
        int offset_new = t + ch * Ldims[0] + batch * Ldims[0] * Ldims[2];
        Lout[offset_new] = Lin[offset++];
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
  if (5 > argc) {
    std::cout << "Usage: " << argv[0] << " <W> <B> <I> <L> [pads=0]\n";
    return -1;
  } else if (5 < argc) {
    pads = atoi(argv[5]);
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

  // read the bias from file
  int Bdims[1]; // {O}
  ifs.open(argv[2]);
  getline(ifs, dummy_line); // dummy line with "O"
  ifs >> Bdims[0];
  int Bsize = Bdims[0];
  if (Bdims[0] != Wdims[2]) {
    std::cout << "Bias should as long as the number of output channels\n";
    return -1;
  }
  float *Bdata = new float[Bsize];
  for (int out_ch = 0; out_ch < Bdims[0]; ++out_ch) {
    ifs >> Bdata[out_ch];
  }
  ifs.close();

  for (int out_ch = 0; out_ch < Wdims[2]; ++out_ch) {
    std::cout << "\033[1mWeights\033[3" << (6 - out_ch) << "m[out_ch=" <<
      out_ch << "] (bias=" << std::setw(7) << std::fixed <<
      std::setprecision(4) << Bdata[out_ch] <<")\033[0m\n";
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

  // read the inputs from file
  int Idims[3]; // {T, B, C}
  ifs.open(argv[3]);
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
    std::cout << "Layout:  {N, C, H, W}\n";
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
    checkCUDA(cudaMalloc(&Wdata_GPU, Wsize * sizeof(float)));
    checkCUDA(cudaMalloc(&Idata_GPU, Isize * sizeof(float)));
    checkCUDA(cudaMalloc(&Odata_GPU, Osize * sizeof(float)));
    if (conv_algos[0].memory) {
      checkCUDA(cudaMalloc(&workspace, conv_algos[0].memory));
    }

    // fill the Odata_GPU with bias
    int bias_expand = Odims4D[0] * Odims4D[3]; // size to expand
    cudaStream_t *cpystreams = new cudaStream_t[bias_expand];
    for (int cnt = 0; cnt < bias_expand; ++cnt) {
      checkCUDA(cudaStreamCreate(&cpystreams[cnt]));
      checkCUDA(cudaMemcpyAsync(Odata_GPU + cnt * Bsize, Bdata,
                                Bsize * sizeof(float), cudaMemcpyHostToDevice,
                                cpystreams[cnt]));
    }
    for (int cnt = 0; cnt < bias_expand; ++cnt) {
      checkCUDA(cudaStreamSynchronize(cpystreams[cnt]));
      checkCUDA(cudaStreamDestroy(cpystreams[cnt]));
    }
    delete[] cpystreams;

    // prepare data on GPU and compute
    float alpha = 1.0;
    float beta = 1.0;
    checkCUDA(cudaMemcpy(Wdata_GPU, Wdata4D, Wsize * sizeof(float),
                         cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(Idata_GPU, Idata, Isize * sizeof(float),
                         cudaMemcpyHostToDevice));
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
    checkCUDA(cudaMemcpy(Odata, Odata_GPU, Osize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    if (conv_algos[0].memory) {
      checkCUDA(cudaFree(workspace));
    }
    checkCUDA(cudaFree(Odata_GPU));

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

    // read the loss from file
    int Ldims[3]; // {T, B, C}
    ifs.open(argv[4]);
    getline(ifs, dummy_line); // dummy line with "T   B   C"
    ifs >> Ldims[0] >> Ldims[1] >> Ldims[2];
    if (Ldims[0] != Odims[0] || Ldims[1] != Odims[1] || Ldims[2] != Odims[2]) {
      std::cout << "Loss is supposed to have the same dimension as outputs\n";
    } else {
      int Lsize = Osize;
      float *Ldata = new float[Lsize];
      for (int batch = 0; batch < Ldims[1]; ++batch) {
        for (int out_ch = 0; out_ch < Ldims[2]; ++out_ch) {
          for (int t = 0; t < Ldims[0]; ++t) {
            int Lcoo[3] = {t, batch, out_ch};
            int offset = offsetTBC(Ldims, Lcoo);
            ifs >> Ldata[offset];
          }
        }
      }
      ifs.close();

      for (int batch = 0; batch < Ldims[1]; ++batch) {
        std::cout << "\033[2mLoss[batch=" << batch << "]\033[0m\n";
        for (int out_ch = 0; out_ch < Ldims[2]; ++out_ch) {
          std::cout << "\033[3" << (6 - out_ch) << "m[out_ch=" <<
            out_ch << "]\033[0m[0:" << Ldims[0] - 1 << "]: ";
          for (int t = 0; t < Ldims[0]; ++t) {
            int Lcoo[3] = {t, batch, out_ch};
            int offset = offsetTBC(Ldims, Lcoo);
            std::cout << std::setw(7) << std::fixed <<
              std::setprecision(4) << Ldata[offset] << "  ";
          }
          std::cout << std::endl;
        }
      }
      std::cout << std::endl;

      // reorganize loss for back propogation
      float *Ldata4D = new float[Osize];
      reorg_loss(Ldims, Ldata, Ldata4D);
      int Ldims4D[4] = {Ldims[1], // batch
                        Ldims[2], // output channels
                        1, // to makeup 4D tensor
                        Ldims[0]}; // time
      int Lstride[4] = {Ldims4D[3] * Ldims4D[2] * Ldims4D[1],
                        Ldims4D[3] * Ldims4D[2],
                        Ldims4D[3], // to satisfy _CHW-packed
                        1}; // next time point is right after

      // backward for bias
      int Bdims4D[4] = {1, Bdims[0], 1, 1};
      int Bstride[4] = {1, 1, 1, 1};

      float *Ldata_GPU = NULL;
      float *Bias_GPU = NULL;
      checkCUDA(cudaMalloc(&Ldata_GPU, Lsize * sizeof(float)));
      checkCUDA(cudaMalloc(&Bias_GPU, Bsize * sizeof(float)));
      checkCUDA(cudaMemcpy(Ldata_GPU, Ldata4D, Lsize * sizeof(float),
                           cudaMemcpyHostToDevice));
      checkCUDA(cudaMemset(Bias_GPU, 0, Bsize * sizeof(float)));

      // create the tensor for loss
      cudnnTensorDescriptor_t loss_desc;
      checkCUDNN(cudnnCreateTensorDescriptor(&loss_desc));
      checkCUDNN(cudnnSetTensorNdDescriptor(loss_desc,             // descriptor
                                            CUDNN_DATA_FLOAT,      // precision
                                            4,                     // dimension
                                            Ldims4D,               // size
                                            Lstride));             // stride
      // create the tensor for bias gradiant
      cudnnTensorDescriptor_t bias_desc;
      checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
      checkCUDNN(cudnnSetTensorNdDescriptor(bias_desc,             // descriptor
                                            CUDNN_DATA_FLOAT,      // precision
                                            4,                     // dimension
                                            Bdims4D,               // size
                                            Bstride));             // stride

      checkCUDNN(cudnnConvolutionBackwardBias(cudnn,
                                              &alpha,
                                              loss_desc,
                                              Ldata_GPU,
                                              &beta,
                                              bias_desc,
                                              Bias_GPU));
      float *Bias = new float[Bsize];
      checkCUDA(cudaMemcpy(Bias, Bias_GPU, Bsize * sizeof(float),
                           cudaMemcpyDeviceToHost));
      checkCUDA(cudaFree(Bias_GPU));
      cudnnDestroyTensorDescriptor(bias_desc);

      std::cout << "\033[1mBias grad\033[0m\n[0:" << Bdims4D[1] - 1 << "]: ";
      for (int out_ch = 0; out_ch < Bdims4D[1]; ++out_ch) {
        std::cout << "\033[3" << (6 - out_ch) << "m" << std::setw(7) <<
          std::fixed << std::setprecision(4) << Bias[out_ch] << "  \033[0m";
      }
      std::cout << std::endl << std::endl;;
      delete[] Bias;

      // backward for weights
      algos_cnt = 6;
      cudnnConvolutionBwdFilterAlgoPerf_t bwd_f_algos[6];
      checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn,
                                                             in_desc,
                                                             loss_desc,
                                                             conv_desc,
                                                             fil_desc,
                                                             algos_cnt,
                                                             &algos_cnt,
                                                             bwd_f_algos));
      for (int i = 0; i < algos_cnt && 0 == bwd_f_algos[i].status; ++i) {
        std::cout << "cudnnConvolutionBwdFilterAlgo_t " <<
          FILTER_BWD_ALGO[bwd_f_algos[i].algo] <<
          " (" << bwd_f_algos[i].time << " ms)\n";
        std::cout << "workspace: " << bwd_f_algos[i].memory << " "
          "determinism: " << bwd_f_algos[i].determinism << " "
          "mathType: " << bwd_f_algos[i].mathType << std::endl;
      }
      std::cout << std::endl;

      workspace = NULL;
      if (bwd_f_algos[0].memory) {
        checkCUDA(cudaMalloc(&workspace, bwd_f_algos[0].memory));
      }
      float *WGdata_GPU;
      checkCUDA(cudaMalloc(&WGdata_GPU, Wsize * sizeof(float)));
      checkCUDNN(cudnnConvolutionBackwardFilter(cudnn,
                                                &alpha,
                                                in_desc,
                                                Idata_GPU,
                                                loss_desc,
                                                Ldata_GPU,
                                                conv_desc,
                                                bwd_f_algos[0].algo,
                                                workspace,
                                                bwd_f_algos[0].memory,
                                                &beta,
                                                fil_desc,
                                                WGdata_GPU));
      if (workspace) {
        checkCUDA(cudaFree(workspace));
        workspace = NULL;
      }
      checkCUDA(cudaMemcpy(Wdata4D, WGdata_GPU, Wsize * sizeof(float),
                           cudaMemcpyDeviceToHost));
      checkCUDA(cudaFree(WGdata_GPU));

      for (int out_ch = 0; out_ch < Wdims4D[0]; ++out_ch) {
        std::cout << "\033[1mWeights gradiant\033[3" << (6 - out_ch) <<
          "m[out_ch=" << out_ch << "]\033[0m\n";
        for (int in_ch = 0; in_ch < Wdims4D[1]; ++in_ch) {
          std::cout << "\033[9" << in_ch + 1 << "m[in_ch=" <<
            in_ch << "]\033[0m[0:" << Wdims4D[3] - 1 << "]: ";
          for (int k = 0; k < Wdims4D[3]; ++k) {
            int offset = k + in_ch * Wdims4D[3] +
              out_ch * Wdims4D[3] * Wdims4D[1];
            std::cout << std::setw(7) << std::fixed <<
              std::setprecision(4) << Wdata4D[offset] << "  ";
          }
          std::cout << std::endl;
        }
      }
      std::cout << std::endl;

      // backward for inputs
      algos_cnt = 6;
      cudnnConvolutionBwdDataAlgoPerf_t bwd_d_algos[6];
      checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnn,
                                                           fil_desc,
                                                           loss_desc,
                                                           conv_desc,
                                                           in_desc,
                                                           algos_cnt,
                                                           &algos_cnt,
                                                           bwd_d_algos));
      for (int i = 0; i < algos_cnt && 0 == bwd_d_algos[i].status; ++i) {
        std::cout << "cudnnConvolutionBwdDataAlgo_t " <<
          DATA_BWD_ALGO[bwd_d_algos[i].algo] <<
          " (" << bwd_d_algos[i].time << " ms)\n";
        std::cout << "workspace: " << bwd_d_algos[i].memory << " "
          "determinism: " << bwd_d_algos[i].determinism << " "
          "mathType: " << bwd_d_algos[i].mathType << std::endl;
      }
      std::cout << std::endl;

      workspace = NULL;
      if (bwd_d_algos[0].memory) {
        checkCUDA(cudaMalloc(&workspace, bwd_d_algos[0].memory));
      }
      float *IGdata_GPU = Idata_GPU; // could reuse
      checkCUDNN(cudnnConvolutionBackwardData(cudnn,
                                              &alpha,
                                              fil_desc,
                                              Wdata_GPU,
                                              loss_desc,
                                              Ldata_GPU,
                                              conv_desc,
                                              bwd_d_algos[0].algo,
                                              workspace,
                                              bwd_d_algos[0].memory,
                                              &beta,
                                              in_desc,
                                              IGdata_GPU));
      if (workspace) {
        checkCUDA(cudaFree(workspace));
        workspace = NULL;
      }
      checkCUDA(cudaMemcpy(Idata, IGdata_GPU, Isize * sizeof(float),
                           cudaMemcpyDeviceToHost));

      for (int batch = 0; batch < Idims[1]; ++batch) {
        std::cout << "\033[1mInputs gradiant[batch=" << batch << "]\033[0m\n";
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

      delete[] Ldata, Ldata4D;
      checkCUDA(cudaFree(Ldata_GPU));
      cudnnDestroyTensorDescriptor(loss_desc);
    } // if(Ldims[] == Odims[])
    checkCUDA(cudaFree(Idata_GPU));
    checkCUDA(cudaFree(Wdata_GPU));
  } // if(isfeasible)

  delete[] Wdata, Bdata, Idata, Odata, Wdata4D;
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyFilterDescriptor(fil_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn);

  return 0;

}
