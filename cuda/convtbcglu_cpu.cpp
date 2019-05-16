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