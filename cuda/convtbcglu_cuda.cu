#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define numThread 1024


template <typename scalar_t>
__global__
void parentConvolve(int inputPlanes, int kernelSize, int outputPlanes, int halfSlide, int batchSize, 
                    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> self,
                    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weight,
                    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias,
					torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output)
{
	int _myID            = threadIdx.x + blockIdx.x * blockDim.x;	                     
	int _myBatch         = _myID          % batchSize;
	int _myOutputSlide   = _myID          / batchSize;
    int _mySlide         = _myOutputSlide % halfSlide;
    int _myOutputChannel = _myOutputSlide / halfSlide;	
	
	if(_myOutputChannel<outputPlanes)
	{
		
		float OutputFront;
		float OutputRear;
		
		OutputFront = bias[_myOutputChannel];
		OutputRear  = OutputFront;
		
		//printf("Hello from Device: %f\n", bias[_myOutputChannel]); 
        // front slide
		
		for(int inputChannel = 0; inputChannel < inputPlanes; inputChannel++)
        {					  
      	  for(int kernel = 0; kernel < kernelSize; kernel++)
      	  {
      		  OutputFront = OutputFront + weight[kernel][inputChannel][_myOutputChannel] * self[_mySlide+kernel][_myBatch][inputChannel];
			  OutputRear = OutputRear + weight[kernel][inputChannel][_myOutputChannel] * self[_mySlide+halfSlide+kernel][_myBatch][inputChannel];
      	  }		   
      	}			  

        // GLU
        output[_mySlide][_myBatch][_myOutputChannel] = 	OutputFront * expf(OutputRear)/(expf(OutputRear)+1);
		//output[_mySlide][_myBatch][_myOutputChannel] = 0.5;
	}
}

extern cudaError_t convtbcglu_cuda(int inputPlanes, int kernelSize, int outputPlanes, int halfSlide, int batchSize, 
                    const torch::Tensor& self,
                    const torch::Tensor& weight,
                    const torch::Tensor& bias,
					torch::Tensor& output)
{
	
	torch::Device deviceCPU(torch::kCPU);
    torch::Device deviceGPU(torch::kCPU);
    cudaError_t error;
	
	if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Run on GPU." << std::endl;
        deviceGPU = torch::Device(torch::kCUDA);
			
		torch::Tensor selfGPU    = self.to(deviceGPU,at::kFloat,false,true);
		torch::Tensor weightGPU  = weight.to(deviceGPU,at::kFloat,false,true);
		torch::Tensor biasGPU    = bias.to(deviceGPU,at::kFloat,false,true);
		torch::Tensor outputGPU  = output.to(deviceGPU,at::kFloat,false,true);
		
		//int numThread = 1024;
	    int numBlock  = (batchSize * halfSlide * outputPlanes + numThread)  / numThread;
	
	    AT_DISPATCH_FLOATING_TYPES(self.type(), "convtbcglu_cuda", ([&] {
			parentConvolve<scalar_t><<<numBlock,numThread>>>(inputPlanes, kernelSize, outputPlanes, halfSlide, batchSize,
	                                       selfGPU.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
										   weightGPU.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), 
										   biasGPU.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(), 
										   outputGPU.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
		
			

		}));
	    
		error=cudaGetLastError();		
		std::cout<<cudaGetErrorString(error) << std::endl;
		
        cudaDeviceSynchronize();
		//output.copy_(outputGPU); //?
		output.copy_(outputGPU);
		cudaFree(selfGPU.data_ptr());
        cudaFree(weightGPU.data_ptr());
        cudaFree(biasGPU.data_ptr());
		cudaFree(outputGPU.data_ptr());
    }
	else
	{
		// CPU
		std::cout << "CUDA is not available! Run on CPU." << std::endl;
		for(int outputChannel = 0; outputChannel < outputPlanes; outputChannel++)
        {
      	  for(int slide = 0; slide < halfSlide; slide++)
      	  {
      
      		  for(int batch = 0; batch < batchSize; batch++)
      		  {
      			  torch::Tensor frontSlide = bias[outputChannel];
      			  torch::Tensor rearSlide  = bias[outputChannel];
      			  // front slide
      		      for(int inputChannel = 0; inputChannel < inputPlanes; inputChannel++)
      		      {					  
      				  int inputStart = slide;
      				  for(int kernel = 0; kernel < kernelSize; kernel++)
      				  {
      					  frontSlide = frontSlide + weight[kernel][inputChannel][outputChannel] * self[inputStart+kernel][batch][inputChannel];
      				  }		   
      		      }			  
      			  // rear slide
      			  for(int inputChannel = 0; inputChannel < inputPlanes; inputChannel++)
      		      {					  
      				  int inputStart = slide+halfSlide;
      				  for(int kernel = 0; kernel < kernelSize; kernel++)
      				  {
      					  rearSlide = rearSlide + weight[kernel][inputChannel][outputChannel] * self[inputStart+kernel][batch][inputChannel];
      				  }		   
      		      }
      			  // applying GLU
      			   output[slide][batch][outputChannel] = frontSlide * exp(rearSlide)/(exp(rearSlide)+1);
      		  }
      	  }	  
        }
	}
	return cudaSuccess;
}

