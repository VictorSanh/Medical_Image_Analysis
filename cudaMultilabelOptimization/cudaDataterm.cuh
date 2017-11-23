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

******************************************************************************

  This software is released under the LGPL license. Details are explained
  in the files 'COPYING' and 'COPYING.LESSER'.
	
*****************************************************************************/

#ifndef DATATERM_CUH
#define DATATERM_CUH

void parzenDataterm(float *dataEnergy, float *image, float *sigmas, float *scribbles, int *numScribbles, int sumScribbles, float rho, float alpha, int width, int height, int nRegions);
void getMinMax(float *data, int width, int height, float *min, float *max);


#endif 
