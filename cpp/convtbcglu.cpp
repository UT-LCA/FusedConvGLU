#include <torch/extension.h>
#include <bits/stdc++.h> 
#include <iostream>

//torch::Tensor convtbcglu_forward(
torch::Tensor convtbcglu_forward(
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
  torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
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
  std::cout << "BIAS" << std::endl;
  std::cout << bias << std::endl;
  auto w_k_len = weight_size[0];
  auto w_i_ch = weight_size[1];
  auto w_o_ch = weight_size[2];
  std::cout << "Weight(" << w_k_len << "," << w_i_ch << "," << w_o_ch << ")\n";
  AT_CHECK(i_c_len == w_i_ch, "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(w_o_ch == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");
  std::cout << weight << std::endl;
/////////////////////////////////
  std::cout << "BEGIN DOING STUFF HERE " << std::endl;
  auto o_t_len = i_t_len-w_k_len+1;   //output tensor time dimension
  auto o_b_len = i_b_len;            //output tensor batch dimension
  auto o_c_len = w_o_ch;             //output tensor channel dimension
  torch::Tensor output= torch::empty({o_t_len/2,o_b_len,o_c_len});  //declare output tensor
  auto self_a = self.accessor<float,3>();           //accessor for input tensor
  auto weight_a = weight.accessor<float,3>();       //accessor for weight tensor
  auto bias_a = bias.accessor<float,1>();           //accessor for bias tensor
//  for(auto i = 0;i<o_t_len;i++){		//cycle through output time
//	for(auto j = 0;j<o_b_len;j++){		// cycle through output batch
//		for(auto k = 0;k<o_c_len;k++){	//cycle through output channel
//			float value = 0;
//			for(auto t = 0;t<w_k_len;t++){    //cycle through weight/kernel size
//				for(auto c = 0;c<w_i_ch;c++){    //cycle through output channels
//					value = value + (float)(weight_a[t][c][k]*self_a[i+t][j][c]);   //accumulate sum
//				}
//			}
//			output[i][j][k] = value + bias_a[0];         
//		}
//	}
// }
	float bias_value = bias_a[0];
	for(auto i = 0;i<o_b_len;i++){	   //bct
		for(auto j = 0;j<o_c_len;j++){
			for(auto k = 0;k<(o_t_len/2);k++){
//				float value = 0;
				float value_1 = 0;
				float value_2 = 0;
				for(auto t = 0;t<w_k_len;t++){
					for(auto c = 0;c<w_i_ch;c++){
						value_1 = value_1 + (float)(weight_a[t][c][j]*self_a[(k)+t][i][c]);
					}
				}
//				output[k][i][j] = value + bias_a[0];
				for(auto t = 0;t<w_k_len;t++){
					for(auto c = 0;c<w_i_ch;c++){
						value_2 = value_2 + (float)(weight_a[t][c][j]*self_a[((o_t_len/2)+k)+t][i][c]);
					}
				}
				output[k][i][j] = (float)(value_1+bias_value) * (float)(1/(1+(float)exp(-1*(value_2+bias_value))));
//				output[2*k][i][j] = value_1 + bias_value;
//				output[2*k+1][i][j] = value_2 + bias_value;
				
				
			}
		}
	}
		//add glu operations
  return output;
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
