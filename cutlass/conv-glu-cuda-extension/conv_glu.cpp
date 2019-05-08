//#include <ATen/ATen.h>
//#include <ATen/NativeFunctions.h>
//#include <tuple>
#include <torch/extension.h>
#include <iostream>

//namespace at {
//namespace native {

int basic_gemm_cuda (torch::Tensor W, torch::Tensor I, torch::Tensor& O);

torch::Tensor conv_glu_forward(const torch::Tensor& self, const torch::Tensor& weight, const torch::Tensor& bias, int64_t pad) {
  AT_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = self.sizes();
  auto weight_size = weight.sizes();

  auto i_t_len = input_size[0];
  auto i_b_len = input_size[1];
  auto i_c_len = input_size[2];
  auto w_o_ch = weight_size[2];
  auto w_k_len = weight_size[0];
  auto olen = i_t_len - w_k_len + 1 + pad * 2;
  auto real_pad = (olen - i_t_len + w_k_len - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  AT_CHECK(i_c_len == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // input * weights + bias -> output_features
  std::cout << "Weights" << std::endl;
  std::cout << weight << std::endl;
  std::cout << "Input" << std::endl;
  std::cout << self << std::endl;
  
  torch::Tensor output = at::empty({
    olen,
    input_size[1],
    weight_size[2],
  }, self.options());
  output.copy_(bias.expand(output.sizes()));
  std::cout << "Output(=bias) before conv_glu" << std::endl;
  std::cout << output << std::endl;
  for (int k = 0; k < w_k_len; k++) {
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    int t = std::min(i_t_len + real_pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      auto W = weight[k];
      std::cout << "weight[k]\n";
      std::cout << W << std::endl;
      auto I = self.narrow(0, iShift, t).view({t * i_b_len, i_c_len});
      auto O = output.narrow(0, oShift, t).view({t * i_b_len, w_o_ch});
      //std::cout << "W" << std::endl;
      //std::cout << W << std::endl;
      //std::cout << "I" << std::endl;
      //std::cout << I << std::endl;
      //std::cout << "O" << std::endl;
      //std::cout << O << std::endl;
      int success = basic_gemm_cuda(W, I, O);  
      if (success == -1) {
      	std::cout << "Assertion error: basic_gemm_cuda is NOT successful!" << std::endl;
      }
      std::cout << "Output after " << k+1 << " loop(s) (starting from loop 0)" <<  std::endl;
      std::cout << O << std::endl;
      //O.addmm_(I, W);
      //O = O + I @ W;
    }
  }
  std::cout << "Output after conv_glu" << std::endl;
  std::cout << output << std::endl;
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_glu_backward(const torch::Tensor& dOutput, const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, int64_t pad) {
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto i_t_len = input_size[0];
  auto i_b_len = input_size[1];
  auto i_c_len = input_size[2];
  auto w_o_ch = weight_size[2];
  auto w_k_len = weight.sizes()[0];
  auto olen = input_size[0] - w_k_len + 1 + pad * 2;
  int real_pad = (olen - i_t_len + w_k_len - 1) / 2;

  torch::Tensor dInput = at::zeros_like(input);
  for (int k = 0; k < w_k_len; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(i_t_len + real_pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * i_b_len, w_o_ch});
      auto dI = dInput.narrow(0, iShift, t).view({t * i_b_len, i_c_len});
      dI.addmm_(dO, weight[k].t());
    }
  }

  torch::Tensor dWeight = at::zeros_like(weight);
  for (int k = 0; k < w_k_len; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(i_t_len + real_pad - k, olen) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * i_b_len, w_o_ch});
      auto I = input.narrow(0, iShift, t).view({t * i_b_len, i_c_len}).t();
      dW.addmm_(I, dO);
    }
  }

  torch::Tensor dBias = at::zeros_like(bias);
  auto tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));

  return std::make_tuple(dInput, dWeight, dBias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_glu_forward, "conv_glu CUDA forward");
  m.def("backward", &conv_glu_backward, "conv_glu CUDA backward");
}
//}
//}
