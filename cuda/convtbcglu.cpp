#include <torch/extension.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern cudaError_t convtbcglu_cuda(int inputPlanes, int kernelSize, int outputPlanes, int halfSlide, int batchSize, 
                    const torch::Tensor& self,
                    const torch::Tensor& weight,
                    const torch::Tensor& bias,
					torch::Tensor& output);

void printSelfTensor(const torch::Tensor& tensortoPrint)
{
	at::IntArrayRef tensorSize = tensortoPrint.sizes();	
	for(int i = 0; i<tensorSize[1];i++)
    {
        std::cout << "Batch " << i << std::endl;
    	for(int j = 0; j<tensorSize[2];j++)
    	{
			std::cout <<"Input Channel " << j << ": ";
    		for(int k = 0; k<tensorSize[0];k++)
    		{
    			std::cout<< std::fixed << std::setprecision(5) << tensortoPrint[k][i][j].item<float>();
    			std::cout <<" ";
    		}
    		std::cout << std::endl;
    	}
    	std::cout << std::endl;
    }
    std::cout << std::endl;	
}

void printOutTensor(const torch::Tensor& tensortoPrint)
{
	at::IntArrayRef tensorSize = tensortoPrint.sizes();	
	for(int i = 0; i<tensorSize[1];i++)
    {
        std::cout << "Batch " << i << std::endl;
    	for(int j = 0; j<tensorSize[2];j++)
    	{
			std::cout <<"Output Channel " << j << ": ";
    		for(int k = 0; k<tensorSize[0];k++)
    		{
    			std::cout<< std::fixed << std::setprecision(5) << tensortoPrint[k][i][j].item<float>();
    			std::cout <<" ";
    		}
    		std::cout << std::endl;
    	}
    	std::cout << std::endl;
    }
    std::cout << std::endl;	
}

void printWeightTensor(const torch::Tensor& tensortoPrint)
{
	at::IntArrayRef tensorSize = tensortoPrint.sizes();	
	for(int i = 0; i<tensorSize[2];i++)
	{
		std::cout << "Output Channel " << i << std::endl;
		for(int j = 0; j<tensorSize[1];j++)
		{
			std::cout << "Input Channel " << j << ": ";
			for(int k = 0; k<tensorSize[0];k++)
			{
				std::cout<< std::fixed << std::setprecision(5) << tensortoPrint[k][j][i].item<float>();
				std::cout <<" ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;	
}

void printBiasTensor(const torch::Tensor& tensortoPrint)
{
	at::IntArrayRef tensorSize = tensortoPrint.sizes();
	
	for(int i = 0; i<tensorSize[0];i++)
	{
		std::cout<< std::fixed << std::setprecision(5) << tensortoPrint[i].item<float>();
		std::cout <<" ";
	}
	std::cout << std::endl;
	std::cout << std::endl;
}

torch::Tensor convtbcglu_forward(
    const torch::Tensor& self,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t pad) {
		
//void convtbcglu_forward() 
//{

  std::cout << "I Love You!" << std::endl;
  
  // Check the input tensors!
  AT_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  AT_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  AT_CHECK(bias.dim() == 1, "Bias must be 1-D");
  
  // Assume that the tensors are inside the GPU
  at::IntArrayRef input_size  = self.sizes();
  at::IntArrayRef weight_size = weight.sizes();
  
  
  //self[data][inputbatch][inputchannel];
  //weigth[data][inputChannel][outputChannel]
  //output[data][outputbatch][outputchannel]
  //bias[outputchannel]
  
  //inputBatch 
  
  int64_t inputLength  = input_size[0];    // data
  int64_t batchSize    = input_size[1];    // batch
  int64_t inputPlanes  = input_size[2];    // inputChannel
  int64_t outputPlanes = weight_size[2];   // outputChannel
  int64_t kernelSize   = weight_size[0];

  int64_t outputLength = inputLength   - kernelSize  + 1 + pad * 2;
  int64_t real_pad     = (outputLength - inputLength + kernelSize - 1) / 2;  

  std::cout << "Input Length : " << inputLength   << std::endl;
  std::cout << "Batch Size   : " << batchSize     << std::endl;
  std::cout << "Input Planes : " << inputPlanes   << std::endl;
  std::cout << "Output Planes: " << outputPlanes  << std::endl;
  std::cout << "kernelSize   : " << kernelSize    << std::endl;
  std::cout << "Output Length: " << outputLength  << std::endl;
  std::cout << "Real Padding : " << real_pad      << std::endl;
  
  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  AT_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  AT_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");
  
  
  // Prepare the output tensor
  torch::Tensor output = at::empty(
                              {
								  outputLength/2,
								  batchSize,
								  outputPlanes,
							   }, 
							   self.options()
							);
   
  // Printing
  std::cout<<"Self Tensor"<<std::endl;
  printSelfTensor(self);

  std::cout<<"Weight Tensor"<<std::endl;
  printWeightTensor(weight);
  
  std::cout<<"Bias Tensor"<<std::endl;
  printBiasTensor(bias);
  
  // Copy the bias to the output
  //output.copy_(bias.expand(output.sizes()));
  std::cout<<"Output Tensor"<<std::endl;
  printOutTensor(output);
  //output[0][0][0] = output[0][0][0] + 0.001;
  //printSelfTensor(output);
  
  int halfSlide = outputLength/2; // it must be guaranted that outputLength is even.
  
  convtbcglu_cuda(inputPlanes, kernelSize, outputPlanes, halfSlide, batchSize,
                  self, weight, bias, output);
  printOutTensor(output);
  
  // Now CUDA //
  
  
  
  /*
  for (int k = 0; k < kernelSize; k++) {
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    int t = std::min(inputLength + real_pad - k, outputLength) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      torch::Tensor W = weight[k]; // Tensor two dimension
      torch::Tensor I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes}); // flattened two-dimensional matrix
      torch::Tensor O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes}); // flattened two-dimensional matrix
      O.addmm_(I, W); // matrix multiplication and addition
    }
	std::cout<<"Iteration: "<< k << std::endl;
	printOutTensor(output);
  }
  */
  
  return output;
  
  
  /*
  torch::Device deviceCPU(torch::kCPU);
  torch::Device deviceGPU(torch::kCPU);
  
  if (torch::cuda::is_available()) 
  {
      std::cout << "CUDA is available! Run on GPU." << std::endl;
      deviceGPU = torch::kCUDA;

  }
  
  test = torch::rand({2, 3});
  torch::Tensor gpu_test = test.to(deviceGPU,at::kFloat,true,true);
  
  torch::Tensor back_test = gpu_test.to(deviceCPU, at::kFloat,true,true);
  
  std::cout << test << std::endl;
  std::cout << gpu_test << std::endl;
  std::cout << back_test << std::endl;
  
  std::cout << self << std::endl;
  std::cout << weight << std::endl;
  std::cout << bias << std::endl;
  
  */
  
  /*
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

  */
  //return output;
}

//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv_tbc_backernelSizeard(
void convtbcglu_backernelSizeard(
    const torch::Tensor& dOutput,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t pad) {

  std::cout << "Hello from convtbcglu_forward()\n";

  //return std::make_tuple(dInput, dWeight, dBias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convtbcglu_forward, "conv_tbc CUDA forward fused with glu");
  m.def("backernelSizeard", &convtbcglu_backernelSizeard, "conv_tbc CUDA backernelSizeard fused with glu");
}
