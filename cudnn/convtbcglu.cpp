#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK
#include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
#include <ATen/cudnn/Handle.h> // for getCudnnHandle

#include <iostream>
#include <string>

std::string CONV_FWD_ALGO[8] = {
  "IMPLICIT_GEMM",
  "IMPLICIT_PRECOMP_GEMM",
  "GEMM",
  "DIRECT",
  "FFT",
  "FFT_TILING",
  "WINOGRAD",
  "WINOGRAD_NONFUSED"};

// Name of function in python module and name used for error messages by
// torch::check* functions.
const char *convtbcglu_forward_name = "convtbcglu_forward";

// Check arguments to convtbcglu_forward
void convtbcglu_forward_check(
    const torch::Tensor& inputs,
    const torch::Tensor& weight,
    const torch::Tensor& convout) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_inputs(inputs, "inputs", 0);
  torch::TensorArg arg_weight(weight, "weight", 1);
  torch::TensorArg arg_convout(convout, "convout", 2);
  // Check arguments. No need to return anything. These functions with throw an
  // error if they fail. Messages are populated using information from
  // TensorArgs.
  torch::checkContiguous(convtbcglu_forward_name, arg_inputs);
  torch::checkScalarType(convtbcglu_forward_name, arg_inputs, torch::kFloat);
  torch::checkBackend(convtbcglu_forward_name, arg_inputs.tensor,
                      torch::Backend::CUDA);
  torch::checkContiguous(convtbcglu_forward_name, arg_weight);
  torch::checkScalarType(convtbcglu_forward_name, arg_weight, torch::kFloat);
  torch::checkBackend(convtbcglu_forward_name, arg_weight.tensor,
                      torch::Backend::CUDA);
  torch::checkContiguous(convtbcglu_forward_name, arg_convout);
  torch::checkScalarType(convtbcglu_forward_name, arg_convout, torch::kFloat);
  torch::checkBackend(convtbcglu_forward_name, arg_convout.tensor,
                      torch::Backend::CUDA);
}

// reorganize weights from KIO to OIK, Win, Wout should no overlap
void reorg_weights(long int *Wdims, float *Win, float *Wout) {
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

torch::Tensor convtbcglu_forward(
    const torch::Tensor& inputs,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int pad = 0) {

  std::cout << "Hello from convtbcglu_forward()\n";
  AT_CHECK(inputs.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = inputs.sizes();
  auto weight_size = weight.sizes();

  auto i_t_len = input_size[0];
  auto i_b_len = input_size[1];
  auto i_c_len = input_size[2];
  std::cout << "Input(" << i_t_len << "," << i_b_len << "," << i_c_len << ")\n";
  std::cout << inputs << std::endl;

  auto w_k_len = weight_size[0];
  auto w_i_ch = weight_size[1];
  auto w_o_ch = weight_size[2];
  std::cout << "Weight(" << w_k_len << "," << w_i_ch << "," << w_o_ch << ")\n";
  std::cout << weight << std::endl;

  std::cout << "Bias(" << bias.size(0) << ")\n";
  std::cout << bias << std::endl;

  AT_CHECK(i_c_len == w_i_ch, "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(w_o_ch == bias.size(0), "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  //////////////////////////////////////
  // prepare data in the right layout //
  //////////////////////////////////////

  int paddings[4] = {0, 0, 0, pad};
  int stride1s[4] = {1, 1, 1, 1};
  int dilation1s[4] = {1, 1, 1, 1}; // normal convolution

  torch::Device device_CPU(torch::kCPU);
  torch::Device device_GPU(torch::kCUDA);
  //torch::Tensor weight_CPU = weight.to(device_CPU, weight.scalar_type(),
  //                                     /*non-blocking=*/false, /*copy=*/true);
  //torch::Tensor weight4D = at::empty({w_o_ch, w_i_ch, 1, w_k_len},
  //                                   weight_CPU.options());
  //long int w_dims[3] = {w_k_len, w_i_ch, w_o_ch};
  //reorg_weights(w_dims, (float*)weight_CPU.data_ptr(),
  //              (float*)weight4D.data_ptr());
  //torch::Tensor weight_GPU = weight4D.to(device_GPU, weight.scalar_type(),
  //                                       false, true);
  torch::Tensor weight_GPU =
    weight.transpose(0, 2).contiguous().view({w_o_ch, w_i_ch, 1, w_k_len});
  if (!weight_GPU.is_cuda()) {
    weight_GPU = weight_GPU.to(device_GPU, weight_GPU.scalar_type(),
                               /*non-blocking=*/false, /*copy=*/true);
  }
  torch::Tensor inputs_GPU = (inputs.is_cuda()) ? inputs.alias() :
    inputs.to(device_GPU, inputs.scalar_type(),
              /*non-blocking=*/false, /*copy=*/true);
  auto o_t_len = i_t_len - w_k_len + 1 + pad * 2;
  torch::Tensor convout_GPU = at::empty({o_t_len, i_b_len, w_o_ch},
                                        inputs.options());
  if (!convout_GPU.is_cuda()) {
    convout_GPU = convout_GPU.to(device_GPU, convout_GPU.scalar_type(),
                                 /*non-blocking=*/false, /*copy=*/false);
  }
  convout_GPU.copy_(bias.expand({o_t_len, i_b_len, w_o_ch}));

  std::cout << "inputs_GPU\n";
  std::cout << inputs_GPU << std::endl;
  std::cout << "weight_GPU\n";
  std::cout << weight_GPU << std::endl;

  /////////////////////////////////////////////////////////
  // To perform convolution with cudnn convolution layer //
  /////////////////////////////////////////////////////////

  // Step 1: Check inputs. This will throw an error if inputs are invalid, so no
  // need to check return codes here.
  convtbcglu_forward_check(inputs_GPU, weight_GPU, convout_GPU);
  // Step 2: Create descriptors
  cudnnHandle_t cudnn_desc = torch::native::getCudnnHandle();
  // Note: 4 is minimum dim for a TensorDescriptor.
  torch::native::TensorDescriptor input_tensor_desc(inputs_GPU, 4);
  input_tensor_desc.set(CUDNN_DATA_FLOAT,
                        {i_b_len, i_c_len, 1, i_t_len},
                        {i_c_len, 1, 1, i_b_len * i_c_len},
                        4);
  std::cout << "input_tensor_desc.print()\n";
  input_tensor_desc.print();

  torch::native::ConvolutionDescriptor conv_desc;
  conv_desc.set(CUDNN_DATA_FLOAT, /*nDim=*/2, paddings,
                stride1s, dilation1s, /*group=*/1);

  // reorganize weights
  torch::native::FilterDescriptor fil_desc;
  fil_desc.set(weight_GPU, 4);

  torch::native::TensorDescriptor output_tensor_desc(convout_GPU, 4);
  output_tensor_desc.set(CUDNN_DATA_FLOAT,
                         {i_b_len, w_o_ch, 1, o_t_len},
                         {w_o_ch, 1, 1, i_b_len * w_o_ch},
                         4);
  std::cout << "output tensor descriptor.print()\n";
  output_tensor_desc.print();

  int algos_cnt = 8;
  cudnnConvolutionFwdAlgoPerf_t conv_algos[8];
  AT_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn_desc,
                                                      input_tensor_desc.desc(),
                                                      fil_desc.desc(),
                                                      conv_desc.desc(),
                                                      output_tensor_desc.desc(),
                                                      algos_cnt,
                                                      &algos_cnt,
                                                      conv_algos));
  bool isfeasible = false;
  for (int i = 0; i < algos_cnt && 0 == conv_algos[i].status; ++i) {
    std::cout << "cudnnConvolutionFwdAlgo_t: " <<
      CONV_FWD_ALGO[conv_algos[i].algo] <<
      " (" << conv_algos[i].time << " ms)\n";
    std::cout << conv_algos[i].memory << " bytes workspace\n";
    std::cout << "determinism: " << conv_algos[i].determinism << " " <<
      " mathType: " << conv_algos[i].mathType << std::endl;
    isfeasible = true;
  }

  if (!isfeasible) {
    // dimensions check
    std::cout << "Padding: {" << paddings[0] << ", " << paddings[1] << "}\n";
    std::cout << "Layout:  {N, H, W, C}\n";
    std::cout << "Wdims4D: {" << weight_GPU.size(0) << ", " <<
      weight_GPU.size(1) << ", " << weight_GPU.size(2) << ", " <<
      weight_GPU.size(3) << "}\n";
    std::cout << "Idims4D: {" << i_b_len << ", " <<
      i_c_len << ", " << 1 << ", " <<
      i_t_len << "}\n";
    std::cout << "Odims4D: {" << i_b_len << ", " <<
      w_o_ch << ", " << 1 << ", " <<
      o_t_len << "}\n";

    int Odims4D[4];
    AT_CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(conv_desc.desc(),
                   input_tensor_desc.desc(), fil_desc.desc(), 4, Odims4D));

    std::cout << "Supposed to be\n";
    std::cout << "Odims4D: {" << Odims4D[0] << ", " << Odims4D[1] <<
      ", " << Odims4D[2] << ", " << Odims4D[3] << "}\n";
  } else {
    char* workspace;
    if (conv_algos[0].memory) {
      cudaMalloc(&workspace, conv_algos[0].memory);
    }
    std::cout << "convout_GPU before covolution\n";
    std::cout << convout_GPU << std::endl;
    float alpha = 1.0;
    float beta = 1.0;
    AT_CUDNN_CHECK(cudnnConvolutionForward(cudnn_desc,
                                           &alpha,
                                           input_tensor_desc.desc(),
                                           inputs_GPU.data_ptr(),
                                           fil_desc.desc(),
                                           weight_GPU.data_ptr(),
                                           conv_desc.desc(),
                                           conv_algos[0].algo,
                                           &workspace,
                                           conv_algos[0].memory,
                                           &beta,
                                           output_tensor_desc.desc(),
                                           convout_GPU.data_ptr()));
    std::cout << "convout_GPU after covolution\n";
    std::cout << convout_GPU << std::endl;

    if (conv_algos[0].memory) {
      cudaFree(workspace);
    }
  }

  ////////////////////
  // To perform GLU //
  ////////////////////
  std::cout << "convout_GPU.narrow().sigmoid_()\n";
  std::cout << convout_GPU.narrow(/*dimension=*/0, /*start=*/o_t_len / 2,
                                  /*length=*/o_t_len / 2).sigmoid_();
  std::cout << std::endl;
  std::cout << "convout_GPU after sigmoid\n";
  std::cout << convout_GPU << std::endl;

  convout_GPU.narrow(0, 0, o_t_len / 2) *=
    convout_GPU.narrow(0, o_t_len / 2, o_t_len / 2);
  std::cout << "convout_GPU after multiplication\n";
  std::cout << convout_GPU << std::endl;

  torch::Tensor outputs = at::empty({o_t_len / 2, i_b_len, w_o_ch},
                                     inputs.options());
  if (outputs.is_cuda()) { // make outputs on the same device as inputs
    outputs = convout_GPU.narrow(0, 0, o_t_len / 2).alias();
  } else {
    outputs = convout_GPU.narrow(0, 0, o_t_len / 2).to(
        device_CPU, outputs.scalar_type(), false, true);
  }

  // They are automatically freed after exit this routine
  //AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_tensor_desc.desc()));
  //AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_tensor_desc.desc()));
  //AT_CUDNN_CHECK(cudnnDestroyFilterDescriptor(fil_desc.desc()));
  //AT_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc.desc()));
  //AT_CUDNN_CHECK(cudnnDestroy(cudnn_desc));

  return outputs;
}

//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_tbc_backward(
void convtbcglu_backward(
    const torch::Tensor& dOutput,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t pad) {

  std::cout << "Hello from convtbcglu_forward()\n";

  //return std::make_tuple(dInput, dWeight, dBias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convtbcglu_forward, "conv_tbc forward fused with glu");
  m.def("backward", &convtbcglu_backward, "conv_tbc backward fused with glu");
}
