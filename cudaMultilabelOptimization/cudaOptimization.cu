/*******************************************************************************************
*    cudaMultilabelOptimization - software to solve multi-label optimization problems      *
*                              Version 1.0		                                           *
*                                                                                          *
*    Copyright 2013 Claudia Nieuwenhuis <claudia.nieuwenhuis@in.tum.de>                    *
********************************************************************************************

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

******************************************************************************/


#ifndef GRADIENTDESCENT_CU
#define GRADIENTDESCENT_CU

#include "cudaOptimization.cuh"
#include <iostream>
#include <cassert>
#include "float.h"
#include "time.h"
#include "cutil.h"

using namespace std;

//cuda specific block sizes
#define BLOCKDIMX 16
#define BLOCKDIMY 16

//maximum number of labels allowed
#define MAXNREGIONS 10


__global__ void getEnergyPrimal(float *gpu_u, float *gpu_dataEnergy, float *gpu_g, float smoothnessWeight, int width, int height, int nRegions, int pitch, float *gpu_energy)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x;

	if(ox >= width) return;

	float en = 0;
	int o, o2;
	float gradx, grady;
	float sum;
	for(int i = 0; i < height; i++)
	{
	    o = i * pitch + ox;
	    sum = 0;
	    for(int n = 0; n < nRegions; n++)
	    {
			o2 = o + n * pitch * height;
			en += gpu_u[o2] * gpu_dataEnergy[o2];

			gradx = 0;
			grady = 0;
			if(ox < width - 1) gradx = gpu_u[o2 + 1] - gpu_u[o2];
			if(i < height - 1) grady = gpu_u[o2 + pitch] - gpu_u[o2];
			sum += sqrt(gradx * gradx + grady * grady);
	    }
	    en += smoothnessWeight * sum * gpu_g[o];
	}
	gpu_energy[ox] = en;
}

__global__ void getEnergyZachDual(float *gpu_u, float *gpu_dataEnergy, float *gpu_xi, int width, int height, int nRegions, int pitch, float *gpu_energy)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x;

	if(ox >= width) return;

	float en = 0;
	float a;
	int o;
	int o2;
	float min;
	float div;
	for(int i = 0; i < height; i++)
	{
	    min = FLT_MAX;
	    div = 0;
	    for(int n = 0; n < nRegions; n++)
	    {
			o = n * pitch * height + i * pitch + ox;
			
			o2 = o + n * pitch * height;
			div = 0;
			if(ox > 0) div += gpu_xi[o2] - gpu_xi[o2 - 1];
			if(i > 0) div += gpu_xi[o2 + pitch * height] - gpu_xi[o2 + pitch * height - pitch];
			a = - div + gpu_dataEnergy[o];
			if(a < min)
			{
				min = a;
			}
	    }
	    en += min;
	}
	gpu_energy[ox] = en;
}

__global__ void getEnergyChambollePrimal(float *gpu_u, float *gpu_dataEnergy, float *gpu_g, int width, int height, int nRegions, int pitch, float *gpu_energy)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x;

	if(ox >= width) return;

	float en = 0;
	int o, o2;
	float gradx, grady;
	float sum;
	for(int i = 0; i < height; i++)
	{
	    o = i * pitch + ox;
	    sum = 0;
	    for(int n = 0; n < nRegions; n++)
	    {
			o2 = o + n * pitch * height;
			en += gpu_u[o2] * gpu_dataEnergy[o2];

			gradx = 0;
			grady = 0;
			if(ox < width - 1) gradx = gpu_u[o2 + 1] - gpu_u[o2];
			if(i < height - 1) grady = gpu_u[o2 + pitch] - gpu_u[o2];
			sum += sqrt(gradx * gradx + grady * grady);
	    }
	    en += sum * gpu_g[o];
	}
	gpu_energy[ox] = en;
}


__global__ void zachChambollePrimal(float *gpu_u, float *gpu_uBar, float *gpu_dataEnergy, float *gpu_xi, int nRegions, int width, int height, int pitch, float stepSize)
{
	//obtain thread index
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int p = y * pitch + x;
	const int m = pitch * height;
	const int n = m * 2;
	int pos;

	const int tx = threadIdx.x + 1;
	const int ty = threadIdx.y + 1;

	float value; 

	__shared__ float2 dualTV[BLOCKDIMX + 1][BLOCKDIMY + 1];
	__shared__ float u[MAXNREGIONS][BLOCKDIMX][BLOCKDIMY];

	// load shared memory

	for(int i = 0; i < nRegions; i++)
	{
		pos = p + i * n;
		if(x < width && y < height)
		{
			u[i][threadIdx.x][threadIdx.y] = gpu_u[p + i * m];
			dualTV[tx][ty].x = gpu_xi[pos];
			dualTV[tx][ty].y = gpu_xi[pos + m];
			if(x == 0)
			{
			    dualTV[tx - 1][ty] = make_float2(0.0f,0.0f);
			}
			else if(threadIdx.x == 0)
			{
			    dualTV[tx - 1][ty].x = gpu_xi[pos - 1];
			    dualTV[tx - 1][ty].y = gpu_xi[pos - 1 + m];
			}

			if(y == 0)
			{
			    dualTV[tx][ty - 1] = make_float2(0.0f,0.0f);
			}
			else if(threadIdx.y == 0)
			{
			    dualTV[tx][ty - 1].x = gpu_xi[pos - pitch];
			    dualTV[tx][ty - 1].y = gpu_xi[pos - pitch + m];
			}

		}

		__syncthreads();

		if(x < width && y < height)
		{
			//compute update step from total variation and data term energy
			value = dualTV[tx][ty].x - dualTV[tx - 1][ty].x + dualTV[tx][ty].y - dualTV[tx][ty - 1].y;
			u[i][threadIdx.x][threadIdx.y] += stepSize * (value - gpu_dataEnergy[p + i * m]);
		}

	}

	if(x >= width || y >= height) return;

	//project to simplex (see paper by Michelot, 1986)
	float sum = 0;
	int ni = 0;
	float a;
	bool b = false;
	while(!b)
	{
		sum = 0;
		ni = 0;
		b = true;
		for(int i = 0; i < nRegions; i++)
		{
			a = u[i][threadIdx.x][threadIdx.y];
			if(a != 0)
			{
				ni++;
				sum += a;
			}
		}
		if(ni)
		{
			a = (sum - 1)/ni;

			for(int i = 0; i < nRegions; i++)
			{
				if(u[i][threadIdx.x][threadIdx.y] != 0)
				{
					u[i][threadIdx.x][threadIdx.y] -= a;
					if(u[i][threadIdx.x][threadIdx.y] < 0)
					{
						b = false;
						u[i][threadIdx.x][threadIdx.y] = 0;
					}
				}
			}
		}
		else
		{
			u[0][threadIdx.x][threadIdx.y] = 1;
		}
	}

	//carry out over relaxation step
	for(int i = 0; i < nRegions; i++)
	{
	    gpu_uBar[p + i * m] = 2 * u[i][threadIdx.x][threadIdx.y] - gpu_u[p + i * m];
	    gpu_u[p + i * m] = u[i][threadIdx.x][threadIdx.y];
	}
}

__global__ void zachDual(float *gpu_u, float *gpu_xi, float *gpu_g, float smoothnessWeight, int nRegions, int width, int height, int pitch, float stepSize)
{
	//compute thread index
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int p = y * pitch + x;
	const int m = pitch * height;
	int pos;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	__shared__ float u[BLOCKDIMX + 1][BLOCKDIMY + 1];
	__shared__ float2 dual[MAXNREGIONS][BLOCKDIMX][BLOCKDIMY];

	// load shared memory
	float sum = 0;
	float dnorm;
	for(int i = 0; i < nRegions; i++)
	{
		pos = p + i * m;
		if(x < width && y < height)
		{
			dual[i][threadIdx.x][threadIdx.y].x = gpu_xi[p + i * 2 * m];
			dual[i][threadIdx.x][threadIdx.y].y = gpu_xi[p + (i * 2 + 1) * m];

			u[tx][ty] =  gpu_u[pos];

			if(x == width-1)
			{
				u[tx + 1][ty] = u[tx][ty];
			}
			else if(tx == blockDim.x - 1)
			{
				u[tx + 1][ty] = gpu_u[pos + 1];
			}
			if(y == height - 1)
			{
				u[tx][ty + 1] = u[tx][ty];
			}
			else if(ty == blockDim.y - 1)
			{
				u[tx][ty+1] = gpu_u[pos + pitch];
			}
		}
		__syncthreads();

		if(x < width && y < height)
		{
			//compute update step
			dual[i][threadIdx.x][threadIdx.y].x += stepSize * (u[tx + 1][ty] - u[tx][ty]);
			dual[i][threadIdx.x][threadIdx.y].y += stepSize * (u[tx][ty + 1] - u[tx][ty]);

			sum += dual[i][threadIdx.x][threadIdx.y].x * dual[i][threadIdx.x][threadIdx.y].x + dual[i][threadIdx.x][threadIdx.y].y * dual[i][threadIdx.x][threadIdx.y].y;
		}
	}

	if(x >= width || y >= height) return;

	//project dual variables by clipping
	sum = sqrtf(sum);
	float edge = gpu_g[p] * smoothnessWeight;

	if(sum > edge)
		dnorm = sum / edge;
	else
		dnorm = 1; 

	for(int i = 0; i < nRegions; i++)
	{
		pos = p + i * 2 * m;
		gpu_xi[pos] = dual[i][threadIdx.x][threadIdx.y].x / dnorm;
		gpu_xi[pos + m] = dual[i][threadIdx.x][threadIdx.y].y / dnorm;
	}
}

//This function is based on the paper
//C. Zach, D. Gallup, J. Frahm and M. Niethammer, Fast global labeling for realtime stereo using multiple plane sweeps, Vision, Modeling and Visualization Workshop (VMV), 2008
//and the formulation used in formula (8) in the paper
//C. Nieuwenhuis and E. Toeppe and D. Cremers, A Survey and Comparison of Discrete and Continuous Multilabel Approaches for the Potts Model, International Journal of Computer Vision, 2013 
void zachPrimalDual(float *u, float *g, float *dataEnergy, float smoothnessWeight, int width, int height, int nRegions, int numSteps, int outStep, bool debugOutput)
{
    if(nRegions > MAXNREGIONS)
    {
		//currently only 14 labels are supported due to memory limitations
		cout << "WARNING: maximum number of regions supported is " << MAXNREGIONS << ", setting to this value" << endl;
		nRegions = MAXNREGIONS;
    }
	
	//Set block size
	dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
	dim3 dimGrid;

	// Set grid size (in number of blocks) for primal dual updates
	dimGrid.x = (width % dimBlock.x) ? (width/dimBlock.x + 1) : (width/dimBlock.x);
	dimGrid.y = (height % dimBlock.y) ? (height/dimBlock.y + 1) : (height/dimBlock.y);

	// Set grid size for energy computation
	dim3 dimBlockEnergy(BLOCKDIMX, 1);	
	dim3 dimGridEnergy( ((width % dimBlock.x) ? (width/dimBlock.x + 1) : (width/dimBlock.x)), 1);

	//allocate memory on GPU
	float *gpu_u, *gpu_uBar, *gpu_xi, *gpu_energy, *gpu_g, *gpu_dataEnergy;
	size_t pitch;
	cutilSafeCall(cudaMallocPitch((void**)&gpu_u, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemset2D(gpu_u, pitch, 0, width * sizeof(float), nRegions * height));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_uBar, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemset2D(gpu_uBar, pitch, 0, width * sizeof(float), nRegions * height));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_xi, &pitch, width * sizeof(float), height * 2 * nRegions));
	cutilSafeCall(cudaMemset2D(gpu_xi, pitch, 0, width * sizeof(float), nRegions * height * 2));
	cutilSafeCall(cudaMalloc((void**)&gpu_energy, sizeof(float) * width));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_g, &pitch, width * sizeof(float), height));
	cutilSafeCall(cudaMemcpy2D(gpu_g, pitch, g, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_dataEnergy, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemcpy2D(gpu_dataEnergy, pitch, dataEnergy, width * sizeof(float), width * sizeof(float), height * nRegions, cudaMemcpyHostToDevice));


	float *energy = new float[width];
	float energyPrimal, energyDual;

	//step sizes chosen according to paper by Pock et al. on preconditioning of primal dual problems, 2011
	float stepSizePrimal = 0.25f;
	float stepSizeDual = 0.5f;

	int step;
	for(step = 0; step < numSteps; step++)
	{
		//dual update
	    zachDual<<< dimGrid, dimBlock >>>(gpu_uBar, 
					      gpu_xi, 
					      gpu_g, 
					      smoothnessWeight,
					      nRegions, 
					      width, 
					      height, 
					      pitch/sizeof(float), 
					      stepSizeDual);

	    cutilSafeCall( cudaThreadSynchronize() );

		//primal update
	    zachChambollePrimal<<< dimGrid, dimBlock >>>(gpu_u, 
						gpu_uBar, 
						gpu_dataEnergy, 
						gpu_xi, 
						nRegions, 
						width, 
						height, 
						pitch/sizeof(float), 
						stepSizePrimal);

	    cutilSafeCall( cudaThreadSynchronize() );

	    if(debugOutput && (!(step % outStep) || (step == numSteps - 1)))
	    {
			//compute primal energy
			getEnergyPrimal<<< dimGridEnergy, dimBlockEnergy >>>(gpu_u, 
										 gpu_dataEnergy, 
										 gpu_g,
									         smoothnessWeight,
										 width, 
										 height, 
										 nRegions, 
										 pitch/sizeof(float), 
										 gpu_energy);

			cutilSafeCall(cudaMemcpy(energy, gpu_energy, width * sizeof(float), cudaMemcpyDeviceToHost) );
			energyPrimal = 0;
			for(int i = 0; i < width; i++)
				energyPrimal += energy[i];

			//compute dual energy
			getEnergyZachDual<<< dimGridEnergy, dimBlockEnergy >>>(gpu_u, 
									   gpu_dataEnergy, 
									   gpu_xi, 
									   width, 
									   height, 
									   nRegions, 
									   pitch/sizeof(float), 
									   gpu_energy);

			energyDual = 0;
			cutilSafeCall(cudaMemcpy(energy, gpu_energy, width * sizeof(float), cudaMemcpyDeviceToHost) );
			for(int i = 0; i < width; i++)
				energyDual += energy[i];

			cout << step << ": Primal energy " << energyPrimal << " Dual energy " << energyDual << " -> primal dual gap: " << energyPrimal - energyDual << endl;
	    }
	}

	//copy result from GPU to CPU
	for(int i = 0; i < nRegions; i++)
	{
	    cutilSafeCall(cudaMemcpy2D(u + i * width * height, width * sizeof(float), gpu_u + i * pitch/sizeof(float) * height, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost));
	}

	//release GPU memory
	cutilSafeCall(cudaFree(gpu_u));
	cutilSafeCall(cudaFree(gpu_uBar));
	cutilSafeCall(cudaFree(gpu_xi));
	cutilSafeCall(cudaFree(gpu_energy));

	delete energy;
}


//This function is based on the paper
//A. Chambolle, D. Cremers, T. Pock, A Convex Approach to Minimal Partitions, SIAM Journal on Imaging Sciences, 2012
//and its formulation used in formula (11) of the paper
//C. Nieuwenhuis and E. Toeppe and D. Cremers, A Survey and Comparison of Discrete and Continuous Multilabel Approaches for the Potts Model, International Journal of Computer Vision, 2013 
__global__ void chambolleDual(float *gpu_u, float *gpu_xi, float *gpu_mu, int nRegions, int width, int height, int pitch, float stepSize)
{
	//obtain thread index
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int p = y * pitch + x;
	const int m = pitch * height;
	int pos;

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	__shared__ float u[BLOCKDIMX + 1][BLOCKDIMY + 1];
	__shared__ float2 dual[MAXNREGIONS][BLOCKDIMX][BLOCKDIMY];

	// load shared memory
	float musum1, musum2;
	for(int i = 0; i < nRegions; i++)
	{
		pos = p + i * m;
		if(x < width && y < height)
		{
			dual[i][threadIdx.x][threadIdx.y].x = gpu_xi[p + i * 2 * m];
			dual[i][threadIdx.x][threadIdx.y].y = gpu_xi[p + (i * 2 + 1) * m];

			u[tx][ty] =  gpu_u[pos];

			if(x == width-1)
			{
				u[tx + 1][ty] = u[tx][ty];
			}
			else if(tx == blockDim.x - 1)
			{
				u[tx + 1][ty] = gpu_u[pos + 1];
			}
			if(y == height - 1)
			{
				u[tx][ty + 1] = u[tx][ty];
			}
			else if(ty == blockDim.y - 1)
			{
				u[tx][ty+1] = gpu_u[pos + pitch];
			}

			//make dual updates
			musum1 = 0;
			musum2 = 0;
			for(int k = 0; k < i; k++)
			{
			    musum1 -= gpu_mu[(k * (nRegions + 1) + i) * m + p];
			    musum2 -= gpu_mu[((nRegions + 1) * (nRegions + 1) + k * (nRegions + 1) + i) * m + p];
			}
			for(int k = i + 1; k <= nRegions; k++)
			{
			    musum1 += gpu_mu[(i * (nRegions + 1) + k) * m + p];
			    musum2 += gpu_mu[((nRegions + 1) * (nRegions + 1) + i * (nRegions + 1) + k) * m + p];
			}
		}
		__syncthreads();

		if(x < width && y < height)
		{
			dual[i][threadIdx.x][threadIdx.y].x += stepSize * (u[tx + 1][ty] - u[tx][ty] + musum1);
			dual[i][threadIdx.x][threadIdx.y].y += stepSize * (u[tx][ty + 1] - u[tx][ty] + musum2);
		}
	}

	if(x >= width || y >= height) return;

	for(int i = 0; i < nRegions; i++)
	{
		pos = p + i * 2 * m;
		gpu_xi[pos] = dual[i][threadIdx.x][threadIdx.y].x;
		gpu_xi[pos + m] = dual[i][threadIdx.x][threadIdx.y].y;
	}
}

__global__ void chambolleMu(float *gpu_mu, float *gpu_muBar, float *gpu_xi, float *gpu_q, int nRegions, int width, int height, int pitch, float stepSize)
{
	//update step for variables mu
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x > width - 1 || y > height * (nRegions + 1) * (nRegions + 1)) return;

	const int m = pitch * height;
	int indexI, indexJ;

	indexI = (y / height) / (nRegions + 1);
	indexJ = (y / height) % (nRegions + 1);
 
	if(indexI >= indexJ) return;
	const int p = y * pitch + x;
	const int o = (nRegions + 1) * (nRegions + 1) * m;

	int pXi = (y % height) * pitch + x;

	float oldMuX = gpu_mu[p];
	float oldMuY = gpu_mu[p + o];

	gpu_mu[p] -= stepSize * (gpu_xi[indexI * 2 * m + pXi] - gpu_xi[indexJ * 2 * m + pXi] - gpu_q[p]);
	gpu_mu[p + o] -= stepSize * (gpu_xi[(indexI * 2 + 1) * m + pXi] - gpu_xi[(indexJ * 2 + 1) * m + pXi] - gpu_q[p + o]);

	gpu_muBar[p] = 2 * gpu_mu[p] - oldMuX;
	gpu_muBar[p + o] = 2 * gpu_mu[p + o] - oldMuY;
}

__global__ void chambolleQ(float *gpu_q, float *gpu_mu, float *gpu_g, float smoothnessWeight, int nRegions, int width, int height, int pitch, float stepSize)
{
	//update step for variables q
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x > width - 1 || y > height * (nRegions + 1) * (nRegions + 1)) return;

	int indexI, indexJ;

	indexI = (y / height) / (nRegions + 1);
	indexJ = (y / height) % (nRegions + 1);
 
	if(indexI >= indexJ) return;
	const int p = y * pitch + x;

	int pXi = (y % height) * pitch + x;
	const int o = pitch * height * (nRegions + 1) * (nRegions + 1);

	float res1, res2;

	res1 = gpu_q[p] - stepSize * gpu_mu[p];
	res2 = gpu_q[p + o] -  stepSize * gpu_mu[p + o];
	
	float t = gpu_g[pXi] * smoothnessWeight;

	float norm = sqrt(res1 * res1 + res2 * res2);
	if(norm > t)
	{
	  res1 *= t/norm;
	  res2 *= t/norm;
	}
	
	gpu_q[p] = res1;
	gpu_q[p + o] = res2;
}

void chambollePrimalDual(float *u, float *g, float *dataEnergy, float smoothnessWeight, int width, int height, int nRegions, int numSteps, int outStep, bool debugOutput)
{
    if(nRegions > MAXNREGIONS)
    {
		cout << "WARNING: maximum number of regions supported is " << MAXNREGIONS << endl;
		nRegions = MAXNREGIONS;
    }

	dim3 dimBlock(BLOCKDIMX, BLOCKDIMY);
	dim3 dimGrid((width % dimBlock.x) ? (width/dimBlock.x + 1) : (width/dimBlock.x), (height % dimBlock.y) ? (height/dimBlock.y + 1) : (height/dimBlock.y)); 
	dim3 dimGridMu;

	dim3 dimBlockEnergy(BLOCKDIMX, 1);
	dim3 dimGridEnergy( ((width % dimBlock.x) ? (width/dimBlock.x + 1) : (width/dimBlock.x)), 1);

	dimGridMu.x = dimGrid.x;
	int nh = height * (nRegions + 1) * (nRegions + 1);
	dimGridMu.y = (nh % dimBlock.y) ? (nh/dimBlock.y + 1) : (nh/dimBlock.y);

	float *gpu_u, *gpu_uBar, *gpu_xi, *gpu_mu, *gpu_muBar, *gpu_q, *gpu_energy, *gpu_g, *gpu_dataEnergy;
	size_t pitch;

	//allocate GPU memory
	cutilSafeCall(cudaMallocPitch((void**)&gpu_u, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemset2D(gpu_u, pitch, 0, width * sizeof(float), nRegions * height));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_uBar, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemset2D(gpu_uBar, pitch, 0, width * sizeof(float), nRegions * height));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_xi, &pitch, width * sizeof(float), height * 2 * (nRegions + 1)));
	cutilSafeCall(cudaMemset2D(gpu_xi, pitch, 0, width * sizeof(float), (nRegions + 1) * height * 2));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_mu, &pitch, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMemset2D(gpu_mu, pitch, 0, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_muBar, &pitch, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMemset2D(gpu_muBar, pitch, 0, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_q, &pitch, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMemset2D(gpu_q, pitch, 0, width * sizeof(float), height * 2 * (nRegions + 1) * (nRegions + 1)));
	cutilSafeCall(cudaMalloc((void**)&gpu_energy, sizeof(float) * width));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_g, &pitch, width * sizeof(float), height));
	cutilSafeCall(cudaMemcpy2D(gpu_g, pitch, g, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMallocPitch((void**)&gpu_dataEnergy, &pitch, width * sizeof(float), height * nRegions));
	cutilSafeCall(cudaMemcpy2D(gpu_dataEnergy, pitch, dataEnergy, width * sizeof(float), width * sizeof(float), height * nRegions, cudaMemcpyHostToDevice));

	float *energy = new float[width];
	float energySum;

	//compute step sizes of all variables based on the paper by Pock et al. on preconditioning of primal dual problems, 2011
    float stepSizeXi = 1.0f/(2 + nRegions);
    float stepSizeu = 1.0f/4;
    float stepSizeMu = 1.0f/3;
    float stepSizeQ = 1;


	int step;
	for(step = 0; step < numSteps; step++)
	{
		//update dual variables xi
	    chambolleDual<<< dimGrid, dimBlock >>>(gpu_uBar, 
							   gpu_xi, 
							   gpu_muBar, 
							   nRegions, 
							   width, height, 
							   pitch/sizeof(float), 
							   stepSizeXi);

	    cutilSafeCall( cudaThreadSynchronize() );

		//update variables q
	    chambolleQ<<< dimGridMu, dimBlock >>>(gpu_q, 
						     gpu_muBar, 
						     gpu_g, 
						     smoothnessWeight,
						     nRegions, 
						     width, height, 
						     pitch/sizeof(float),
						     stepSizeQ);

	    cutilSafeCall( cudaThreadSynchronize() );

		//update primal variables u
	    zachChambollePrimal<<< dimGrid, dimBlock >>>(gpu_u,
						gpu_uBar,
						gpu_dataEnergy,
						gpu_xi, 
						nRegions, 
						width, height, 
						pitch/sizeof(float), 
						stepSizeu);
		
	    cutilSafeCall( cudaThreadSynchronize() );

		//update variables mu
	    chambolleMu<<< dimGridMu, dimBlock >>>(gpu_mu,
						      gpu_muBar,
						      gpu_xi, 
						      gpu_q, 
						      nRegions, 
						      width, height, 
						      pitch/sizeof(float), 
						      stepSizeMu);

	    cutilSafeCall( cudaThreadSynchronize() );

	    if(debugOutput && (!(step % outStep) || (step == numSteps - 1)))
	    {
			//since primal dual gap cannot be computed compute only primal energy
			getEnergyPrimal<<< dimGridEnergy, dimBlockEnergy >>>(gpu_u, 
										 gpu_dataEnergy, 
										 gpu_g, 
 									         smoothnessWeight,
										 width, 
										 height, 
										 nRegions, 
										 pitch/sizeof(float), 
										 gpu_energy);

			cutilSafeCall(cudaMemcpy(energy, gpu_energy, width * sizeof(float), cudaMemcpyDeviceToHost) );
			energySum = 0;
			for(int i = 0; i < width; i++)
				energySum += energy[i];

			cout << step << ": primal energy " << energySum << endl;

	    }

	}

	//copy result from GPU to CPU
	for(int i = 0; i < nRegions; i++)
	{
	    cutilSafeCall(cudaMemcpy2D(u + i * width * height, width * sizeof(float), gpu_u + i * pitch/sizeof(float) * height, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost));
	}

	//release GPU memory
	cutilSafeCall(cudaFree(gpu_u));
	cutilSafeCall(cudaFree(gpu_uBar));
	cutilSafeCall(cudaFree(gpu_mu));
	cutilSafeCall(cudaFree(gpu_muBar));
	cutilSafeCall(cudaFree(gpu_q));
	cutilSafeCall(cudaFree(gpu_xi));
	cutilSafeCall(cudaFree(gpu_energy));
	delete energy;
}





#endif 
