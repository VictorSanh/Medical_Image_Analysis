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

#ifndef _main_C
#define _main_C


#include "CImg.h"
#include "imageSegmentation.cpp"
#include "dataterm.cpp"
#include "dice.cpp"
#include "automaticValueExploration.cpp"
#include "segmentationManipulation.cpp"
#include "params.h"
#include <iostream>


int main()
{
    //load parameters from parameter file 'parameters.txt'
    CParams<float> params;
    CImg<float> *scribbleMap = NULL, *img = NULL;
    params.readParams("parameters.txt");
    if(params.debugOutput) cout << "Parameters read" << endl;

    //read image and normalize to range [0,255]
    img = new CImg<float>(params.imageFile.c_str());
    float imgMax = img->max();
    *img = *img / imgMax * 255;
   
    //initialize data term
    if(params.debugOutput) cout << "Creating dataterm" << endl;
    Dataterm<float> *dataterm = new SpatiallyVaryingParzenDataterm<float>(params.colorVariance, params.scribbleDistanceFactor);

    //initialize segmentation
    if(params.debugOutput) cout << "Initialize cuda for segmentation" << endl;
    ImageSegmentation<float> segmentation(params.debugOutput);

    bool interactive;
    //read scribble file if indicated
    if(params.scribbleFile != "")
    {
		scribbleMap = new CImg<float>(params.scribbleFile.c_str());
		cout << params.scribbleFile.c_str() <<endl;
		if(params.debugOutput) cout << "Loading scribble map" << endl;

		if(scribbleMap->width() != img->width() || scribbleMap->height() != img->height())
		{
		    cout << "WARNING: scribble file size does not match image size" << endl;
		}

		//read number of regions from scribble file
		interactive = false;
		int m = scribbleMap->max();
		params.nRegions = m + 1;
    }
    else
    { 
		//initialize an empty scribble map with -1
		interactive = true;
		scribbleMap = new CImg<float>(img->width(), img->height(), 1, 1, -1);
    }

    //execute segmentation algorithm
    
    //with user interaction
    
    //OutputName Processing for more visibility
    string outputName = params.imageFile.substr(0, params.imageFile.size()-4);
    outputName = outputName.substr(outputName.find("/") + 1); 
    
    int  edgeVar = 5;
    char outputParameters[255];
    sprintf(outputParameters, "%0.1f_%0.3f_%i_%0.1f", params.colorVariance, params.smoothnessWeight, edgeVar, params.scribbleDistanceFactor);
    string outputNameWithParameters = outputName + outputParameters;
    
    if(interactive)
	{
		
		segmentation.executeInteractive(dataterm, scribbleMap, img, params.brushSize, params.smoothnessWeight, params.debugOutput, params.optimizationMethod, params.resultsFolder, outputName, params.numSteps, params.brushDensity, params.outputEveryNSteps);
	}
    else //use loaded scribble map
    {	
	dataterm->readScribblesFromMap(scribbleMap, img);
	dataterm->computeDataEnergy(img);
	segmentation.executeAutomatic(&(dataterm->dataEnergy), 
				      img, 
				      params.smoothnessWeight, 
				      //lambdaToTest[k],
				      params.debugOutput, 
				      params.optimizationMethod, 
				      params.numSteps, 
				      params.outputEveryNSteps);
	segmentation.drawAndSaveResults(img, 
					scribbleMap, 
					&dataterm->dataEnergy, 
					params.resultsFolder, 
					outputNameWithParameters);
	//segmentation.segmentation.display();
    }

    
    /*CImg<float> *estimated = new CImg<float>("Outputs/u/flowers1.3_125.000_5_5.0.cimg");
    CImg<float> *groundTruth = new CImg<float>("Outputs/u/flowersMorePreciseScribble1.3_125.000_5_5.0.cimg");
    std::list<float> scores(1+(*groundTruth).max());
    diceScore((*estimated), (*groundTruth), scores);
    cout << "Average Dice Score: " << averageDiceScore(scores) <<endl;
    delete estimated;
    estimated = NULL;
    delete groundTruth;
    groundTruth = NULL;*/
    
    //release memory
    if(img)
    {
		delete img;
		img = NULL;
    }
    if(dataterm) 
    {
		delete dataterm;
		dataterm = NULL;
    }
    if(scribbleMap)
    {
		delete scribbleMap;
		scribbleMap = NULL;
    }
    return 0;
}

/*int main()
{
  testValues();
  return 0;
}*/

#endif
