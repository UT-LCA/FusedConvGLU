#include <torch/extension.h>

#include <iostream>

//torch::Tensor convtbcglu_forward(
void convtbcglu_forward(
    const torch::Tensor& self,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t pad) {

  std::cout << "Hello from convtbcglu_forward()\n";
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
  std::cout << "Input(" << i_t_len << "," << i_b_len << "," << i_c_len << ")\n";
  std::cout << self << std::endl;
  std::cout << "Input[0]:\n";
  std::cout << self[0] << std::endl;
  std::cout << "Input[2][1][0]:\n";
  std::cout << self[2][1][0] << std::endl;

  auto w_k_len = weight_size[0];
  auto w_i_ch = weight_size[1];
  auto w_o_ch = weight_size[2];
  std::cout << "Weight(" << w_k_len << "," << w_i_ch << "," << w_o_ch << ")\n";
  AT_CHECK(i_c_len == w_i_ch, "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(w_o_ch == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");
  std::cout << weight << std::endl;

  //return output;
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
