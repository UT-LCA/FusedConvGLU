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
  
  return output;
  
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> convtbcglu_backernelSizeard(
const torch::Tensor& dOutput, 
const torch::Tensor& input, 
const torch::Tensor& weight, 
const torch::Tensor& bias, 
int64_t pad) 
{
  at::IntArrayRef input_size  = input.sizes();
  at::IntArrayRef weight_size = weight.sizes();

  int inputLength      = input_size[0];
  int batchSize        = input_size[1];
  int inputPlanes      = input_size[2];
  int outputPlanes     = weight_size[2];
  int kernelSize       = weight.sizes()[0];
  int outputLength     = input_size[0] - kernelSize + 1 + pad * 2;
  int real_pad         = (outputLength - inputLength + kernelSize - 1) / 2;

  std::cout << "Input Length : " << inputLength   << std::endl;
  std::cout << "Batch Size   : " << batchSize     << std::endl;
  std::cout << "Input Planes : " << inputPlanes   << std::endl;
  std::cout << "Output Planes: " << outputPlanes  << std::endl;
  std::cout << "kernelSize   : " << kernelSize    << std::endl;
  std::cout << "Output Length: " << outputLength  << std::endl;
  std::cout << "Real Padding : " << real_pad      << std::endl;
  
   // Printing
  std::cout<<"Before backward" << std::endl;
  std::cout<<"dOutput Tensor"<<std::endl;
  printOutTensor(dOutput);
  
  std::cout<<"input Tensor"<<std::endl;
  printSelfTensor(input);

  std::cout<<"Weight Tensor"<<std::endl;
  printWeightTensor(weight);
  
  std::cout<<"Bias Tensor"<<std::endl;
  printBiasTensor(bias);
  
  torch::Tensor dInput = at::zeros_like(input);
  for (int k = 0; k < kernelSize; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(inputLength + real_pad - k, outputLength) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      torch::Tensor dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      torch::Tensor dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  torch::Tensor dWeight = at::zeros_like(weight);
  for (int k = 0; k < kernelSize; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    int t = std::min(inputLength + real_pad - k, outputLength) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      torch::Tensor dW = dWeight[k];
      torch::Tensor dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      torch::Tensor I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  torch::Tensor dBias = at::zeros_like(bias);
  torch::Tensor tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));

  std::cout<<"After backward" << std::endl;
  
  std::cout<<"input Tensor"<<std::endl;
  printSelfTensor(dInput);

  std::cout<<"Weight Tensor"<<std::endl;
  printWeightTensor(dWeight);
  
  std::cout<<"Bias Tensor"<<std::endl;
  printBiasTensor(dBias);
  
  return std::make_tuple(dInput, dWeight, dBias);
  
  
  
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convtbcglu_forward, "conv_tbc CUDA forward fused with glu");
  m.def("backward", &convtbcglu_backernelSizeard, "conv_tbc CUDA backward fused with glu");
}
