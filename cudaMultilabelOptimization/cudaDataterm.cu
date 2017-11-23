/*********************************************************************************
*    cudaSegmentation - software to solve multi-label optimization problems      *
*                              Version 1.0		                                 *
*                                                                                *
*    Copyright 2013 Claudia Nieuwenhuis <claudia.nieuwenhuis@in.tum.de>          *
**********************************************************************************

  If you use this software for research purposes, YOU MUST CITE the corresponding
  of the following papers in any resulting publication:

	If you use the spatially varying data term implemented in dataterm.cpp and
	cudaDataterm.cu: [1]
	
	If you use any other of the segmentation or optimization routines: [2]

	In addition, if you use the optimization routine zachPrimalDual: [3],[5]

	In addition, if you use the optimization routine chambollePrimalDual: [4],[5]


    [1] C. Nieuwenhuis and D. Cremers, 
		Spatially Varying Color Distributions for Interactive Multi-Label 
		Segmentation, 
		Transactions on Pattern Analysis and Machine Intelligence, 2013		

	[2] C. Nieuwenhuis and E. Toeppe and D. Cremers, 
		A Survey and Comparison of Discrete and Continuous Multilabel 
		Approaches for the Potts Model, 
		International Journal of Computer Vision, 2013     

	[3] C. Zach, D. Gallup, J. Frahm and M. Niethammer, 
		Fast global labeling for realtime stereo using multiple plane sweeps, 
		Vision, Modeling and Visualization Workshop (VMV), 2008

	[4] A. Chambolle, D. Cremers, T. Pock, 
		A Convex Approach to Minimal Partitions, 
		SIAM Journal on Imaging Sciences, 2012

	[5] T. Pock, D. Cremers, H. Bischof, A. Chambolle, 
		An Algorithm for Minimizing the Piecewise Smooth Mumford-Shah Functional, 
		ICCV 2009

******************************************************************************

  This software is released under the LGPL license. Details are explained
  in the files 'COPYING' and 'COPYING.LESSER'.
	
*****************************************************************************/

#ifndef dataterm_CU
#define dataterm_CU


#include "cudaDataterm.cuh"
#include <iostream>
#include <cassert>
#include "float.h"
#include "cutil.h"

#define BLOCKDIMX 16
#define BLOCKDIMY 16

using namespace std;

texture<float, 1, cudaReadModeElementType> texScribbles;
texture<int, 1, cudaReadModeElementType> texNumScribbles;
cudaChannelFormatDesc texDescScribbles = cudaCreateChannelDesc<float> ();
cudaChannelFormatDesc texDescNumScribbles = cudaCreateChannelDesc<int> ();

//convert data likelihood into energy by taking negative logarithm and handling special cases
__global__ void setEnergyGPU(float *gpu_dataEnergy, int width, int height, int pitch, float min, float max)
{
	float eps = 0.0000001f; //very close to 0 for normalization to avoid log(0) at the end of this function

 	int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int oy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ox >= width || oy >= height)
		return;

	int o = oy * pitch + ox;

	if(gpu_dataEnergy[o] == -1) //hard constraint for belonging to current region -> low energy
	{
		gpu_dataEnergy[o] = 0;
	}
	else if(gpu_dataEnergy[o] == -2) //hard constraint for not belonging to current region -> high energy
	{
		gpu_dataEnergy[o] = 1000;
	}
	else //normalize energy to the interval ]0,1[ and take its logarithm
	{
	    if(min != max)
		gpu_dataEnergy[o] = -log((gpu_dataEnergy[o] - min) / (max - min) * (1 - eps) + eps); 
	    else 
		gpu_dataEnergy[o] = -log(gpu_dataEnergy[o] + eps);
	}
}

//get minimum and maximum value of data image fast
__global__ void getMinMaxGPU(float *data, int width, int height, int pitch, float *gpu_min, float *gpu_max)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x;

	if (ox >= width)
		return;

	float min = FLT_MAX;
	float max = -FLT_MAX;

	for (int y = 0; y < height; y++)
	{
		int o = y * pitch + ox;
		if (data[o] < min && data[o] >= 0)
			min = data[o];
		if (data[o] > max && data[o] >= 0)
			max = data[o];
	}

	gpu_min[ox] = min;
	gpu_max[ox] = max;
}


//compute data energy (core routine on GPU)
__global__ void getData(float *gpu_image, float *gpu_dataEnergy, int width, int height, int nRegions, float colorVariance, float scribbleDistanceFactor, int pitch, float *gpu_rhos, int sumScribbles)
{
	// Global thread location
	int ox = blockDim.x * blockIdx.x + threadIdx.x;
	int oy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ox >= width || oy >= height)
		return;

	int o = oy * pitch + ox;

	for (int n = 0; n < nRegions; n++)
	{
	    if(tex1Dfetch(texNumScribbles, n) == 0) //region without scribbles is assigned infinity data cost in setEnergyGPU function
	    {
			gpu_dataEnergy[o + n * pitch * height] = -2; 
	    }
	    else
			gpu_dataEnergy[o + n * pitch * height] = 0;
	}

	float d, dist;
	float a, b;
	int t = 0;
	bool hardconstraint = false;
	
	//if spatial variance is used (i.e. scribbleDistanceFactor < infinity) find distance of current pixel to the closest scribble in each region and save in gpu_rhos (= variance of the spatial Gaussian)
	if(scribbleDistanceFactor < 1000)
	{
	    for (int n = 0; n < nRegions; n++)
	    {
			if(tex1Dfetch(texNumScribbles, n) > 0)
			{
				dist = FLT_MAX;
				for (int i = 0; i < tex1Dfetch(texNumScribbles, n); i++)
				{
					a = (float) (ox - tex1Dfetch(texScribbles, t));
					b = (float) (oy - tex1Dfetch(texScribbles, t + 1));
					d = a * a + b * b;

					if(d < dist)
					{
						dist = d;
					}
					t += 5;
				}
				gpu_rhos[o + n * pitch * height] = scribbleDistanceFactor * sqrt(dist);
					
				if(dist == 0) //dist = 0 indicates that current location lies on a scribble -> spatial Gaussian with variance 0 -> assign pixel to correct region as hard constraint
				{
					hardconstraint = true; 
					for (int h = 0; h < nRegions; h++)
					{
						gpu_dataEnergy[o + h * pitch * height] = -2; //indicates pixel is not in region h (used in setEnergyGPU function)
					}
					gpu_dataEnergy[o + n * pitch * height] = -1; // indicates pixel is in region n
					break;
				}
			}
	    }
	}
	if(hardconstraint) return; 
	//otherwise compute spatial and color kernel

	float r = 1;
	float rho, colorkernel, spacekernel, normalization;

	//precalculated factors for color Gaussian
	float c1 = powf(2 * 3.14159 * colorVariance * colorVariance, -1.5f);
	float c2 = -2 * colorVariance * colorVariance;
	float c3, c4;


	t = 0;
	for (int n = 0; n < nRegions; n++)
	{
	    if(tex1Dfetch(texNumScribbles, n) > 0)
	    {
			//set spatial variance rho
			normalization = 0;
			if(scribbleDistanceFactor < 1000) //spatial variance is used, i.e. scribbleDistanceFactor < infinity
				rho = gpu_rhos[o + n * pitch * height];
			else rho = 1; //only color kernel is used

			c3 = 1/(rho * sqrtf(2 * 3.14159));
			c4 = -2 * rho * rho;


			//color kernel
		    
			for (int i = 0; i < tex1Dfetch(texNumScribbles, n); i++)
			{
				if(colorVariance < 1000) //if color kernel is used, i.e. color variance colorVariance < infinity compute color Gaussian
				{
					r = 0;
					for(int c = 0; c < 3; c++) //RGB distance
					{
						a = (gpu_image[o + c * pitch * height] - tex1Dfetch(texScribbles, t + 2 + c));
						r += a * a;
					}
					colorkernel = exp(r / c2);
				}
				else
				{
					c1 = 1;
					colorkernel = 1;
				}
				
				//spatial kernel
				if (scribbleDistanceFactor < 1000) //if spatial kernel is used
				{
					a = (float) (ox - tex1Dfetch(texScribbles, t));
					b = (float) (oy - tex1Dfetch(texScribbles, t + 1));
					r = a * a + b * b;

					//precalculated factors for spatial Gaussian
					spacekernel = exp(r / c4);
				}
				else
				{
					spacekernel = 1;
					c3 = 1;
				}

				gpu_dataEnergy[o + n * pitch * height] += colorkernel * spacekernel;
				normalization += spacekernel;
				t += 5;
			}

			gpu_dataEnergy[o + n * pitch * height] *= c1 * c3;
			normalization *= c3;

			gpu_dataEnergy[o + n * pitch * height] /= tex1Dfetch(texNumScribbles, n);
	    }
	}
}


//*****compute data energy
//dataEnergy: return parameter
//image: image to compute the data term for
//rhos: return parameter containing the resulting spatial variance at each pixel computed in this function (corresponding to closest distance to scribble point of this region, see paper)
//scribbles: scribble points (x and y coordinate and corresponding RGB data) saved in format xyrgbxyrgb...
//numScribbles: number of scribble points for each of the regions
//sumScribbles: total number of scribble points
//colorVariance: user indicated color variance for color density estimation in Parzen density
//scribbleDistanceFactor: factor for increasing spatial variance in Parzen density, called alpha in the paper
//width: width of image
//height: height of image
//nRegions: number of regions/segments in the segmentation
//***************************

 
void parzenDataterm(float *dataEnergy, float *image, float *rhos, float *scribbles, int *numScribbles, int sumScribbles, float colorVariance, float scribbleDistanceFactor, int width, int height, int nRegions)
{
    float *gpu_rhos, *gpu_image, *gpu_scribbles, *gpu_dataMin, *gpu_dataMax, *gpu_dataEnergy;
    int *gpu_numScribbles;
    size_t pitch;

    cutilSafeCall(cudaMallocPitch((void**) &gpu_rhos, &pitch, width * sizeof(float), height * nRegions));

    cutilSafeCall(cudaMallocPitch((void**) &gpu_image, &pitch, width * sizeof(float), height * 3));
    cutilSafeCall(cudaMemcpy2D(gpu_image, pitch, image, width * sizeof(float), width * sizeof(float), height * 3, cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMalloc(&gpu_scribbles, sumScribbles * sizeof(float) * 5));
    cutilSafeCall(cudaMemcpy(gpu_scribbles, scribbles, sumScribbles * sizeof(float) * 5, cudaMemcpyHostToDevice));
    
    cutilSafeCall(cudaMalloc((void**)&gpu_numScribbles, nRegions * sizeof(int)));
    cutilSafeCall(cudaMemcpy(gpu_numScribbles, numScribbles, nRegions * sizeof(int), cudaMemcpyHostToDevice));

    cutilSafeCall(cudaMalloc(&gpu_dataMin, sizeof(float) * width));
    cutilSafeCall(cudaMalloc(&gpu_dataMax, sizeof(float) * width));

    cutilSafeCall(cudaMallocPitch((void**) &gpu_dataEnergy, &pitch, width * sizeof(float), height * nRegions));

    texScribbles.addressMode[0] = cudaAddressModeClamp;
    texScribbles.addressMode[1] = cudaAddressModeClamp;
    texScribbles.filterMode = cudaFilterModePoint;
    texScribbles.normalized = false;
    cutilSafeCall(cudaBindTexture(0, &texScribbles, gpu_scribbles, &texDescScribbles, sumScribbles * 5 * sizeof(float)));

    texNumScribbles.addressMode[0] = cudaAddressModeClamp;
    texNumScribbles.addressMode[1] = cudaAddressModeClamp;
    texNumScribbles.filterMode = cudaFilterModePoint;
    texNumScribbles.normalized = false;
    cutilSafeCall(cudaBindTexture(0, &texNumScribbles, gpu_numScribbles, &texDescNumScribbles, nRegions * sizeof(int)));

//compute unnormalized data energy
    dim3 dimGrid, dimBlock(BLOCKDIMX, 1);

    dimGrid.x = (width % dimBlock.x) ? (width / dimBlock.x + 1) : (width / dimBlock.x);
    dimGrid.y = (height % dimBlock.y) ? (height / dimBlock.y + 1) : (height / dimBlock.y);

    getData<<< dimGrid, dimBlock >>>(gpu_image, gpu_dataEnergy, width, height, nRegions, colorVariance, scribbleDistanceFactor, pitch/sizeof(float), gpu_rhos, sumScribbles);

	//find minimum and maximum energy value for normalization in setEnergyGPU
    dim3 dimGridMinMax, dimBlockMinMax(BLOCKDIMX, 1);

    dimGridMinMax.x = (width % dimBlockMinMax.x) ? (width / dimBlockMinMax.x + 1) : (width / dimBlockMinMax.x);
    dimGridMinMax.y = 1;

    float *minn = new float[width];
    float *maxx = new float[width];
    float min, max;

    min = FLT_MAX;
    max = -FLT_MAX;

    for (int i = 0; i < nRegions; i++)
    {
		getMinMaxGPU<<< dimGridMinMax, dimBlockMinMax >>>(gpu_dataEnergy + i * pitch/sizeof(float) * height, width, height, pitch/sizeof(float), gpu_dataMin, gpu_dataMax);

		cutilSafeCall(cudaMemcpy(minn, gpu_dataMin, sizeof(float) * width, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(maxx, gpu_dataMax, sizeof(float) * width, cudaMemcpyDeviceToHost));

		for (int j = 0; j < width; j++)
		{
			if (minn[j] < min)
			min = minn[j];
			if (maxx[j] > max)
			max = maxx[j];
		}
    }

    for (int i = 0; i < nRegions; i++)
    {
	setEnergyGPU <<< dimGrid, dimBlock >>>(gpu_dataEnergy + i * pitch/sizeof(float) * height, width, height, pitch/sizeof(float), min, max);
    }
    delete minn;
    delete maxx;

	//copy result to CPU
    cutilSafeCall(cudaMemcpy2D(dataEnergy, width * sizeof(float), gpu_dataEnergy, pitch, width * sizeof(float), height * nRegions, cudaMemcpyDeviceToHost));

    // release GPU memory
    cutilSafeCall(cudaFree(gpu_image));
    cutilSafeCall(cudaFree(gpu_rhos));
    cutilSafeCall(cudaFree(gpu_numScribbles));
    cutilSafeCall(cudaUnbindTexture(texScribbles));
    cutilSafeCall(cudaUnbindTexture(texNumScribbles));
    cutilSafeCall(cudaFree(gpu_dataMin));
    cutilSafeCall(cudaFree(gpu_dataMax));
}




#endif /* CUDATEST_H_ */

