#ifndef _automaticValueExploration_C
#define _automaticValueExploration_C

#include "CImg.h"
#include "imageSegmentation.cpp"
#include "dataterm.cpp"
#include "params.h"
#include <iostream>


int testValues()
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
    
    //read scribble file if indicated
    if(params.scribbleFile != "")
    {
	scribbleMap = new CImg<float>(params.scribbleFile.c_str());
	if(params.debugOutput) cout << "Loading scribble map" << endl;

	if(scribbleMap->width() != img->width() || scribbleMap->height() != img->height())
	{
	    cout << "WARNING: scribble file size does not match image size" << endl;
	}

	//read number of regions from scribble file
	int m = scribbleMap->max();
	params.nRegions = m + 1;
    }
    else
    { 	
	cout << "NO SCRIBBLE FILE FILLED" <<endl;
    }
    
    
    
    
    
    //Parameters to test
    float colorVarianceList[1] = {1.3}; //sigma
    float scribbleDistanceFactorList[16] = {0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 8, 10, 50, 100, 1000, 10000}; //alpha
    float smoothnessWeightList[1] = {100}; //lambda
    
    
    
    
    for (int sigma=0; sigma<int(sizeof(colorVarianceList)/sizeof(colorVarianceList[0])); sigma++){
      params.colorVariance = colorVarianceList[sigma];
      for (int alpha=0; alpha<int(sizeof(scribbleDistanceFactorList)/sizeof(scribbleDistanceFactorList[0])); alpha++){
	params.scribbleDistanceFactor = scribbleDistanceFactorList[alpha];
	for (int lambda=0; lambda<int(sizeof(smoothnessWeightList)/sizeof(smoothnessWeightList[0])); lambda++){
	    params.smoothnessWeight = smoothnessWeightList[lambda];

	    
	    //initialize data term
	    if(params.debugOutput) cout << "Creating dataterm" << endl;
	    Dataterm<float> *dataterm = new SpatiallyVaryingParzenDataterm<float>(params.colorVariance, params.scribbleDistanceFactor);
	    //initialize segmentation
	    if(params.debugOutput) cout << "Initialize cuda for segmentation" << endl;
	    ImageSegmentation<float> segmentation(params.debugOutput);




	    //OutputName Processing for more visibility
	    string outputName = params.imageFile.substr(0, params.imageFile.size()-4);
	    outputName = outputName.substr(outputName.find("/") + 1); 
	    
	    int  edgeVar = 5;
	    char outputParameters[255];
	    sprintf(outputParameters, "%0.1f_%0.3f_%i_%0.1f", params.colorVariance, params.smoothnessWeight, edgeVar, params.scribbleDistanceFactor);
	    string outputNameWithParameters = outputName + outputParameters;
	    

		
	    dataterm->readScribblesFromMap(scribbleMap, img);
	    dataterm->computeDataEnergy(img);
	    segmentation.executeAutomatic(&(dataterm->dataEnergy), 
					  img, 
					  params.smoothnessWeight, 
					  params.debugOutput, 
					  params.optimizationMethod, 
					  params.numSteps, 
					  params.outputEveryNSteps);
	    segmentation.drawAndSaveResults(img, 
					    scribbleMap, 
					    &dataterm->dataEnergy, 
					    params.resultsFolder, 
					    outputNameWithParameters);
	    
	    
	    if(dataterm) 
	    {
		delete dataterm;
		dataterm = NULL;
	    }
	}
      }
    }
    
    if(img)
    {
	delete img;
	img = NULL;
    }
    if(scribbleMap)
    {
	delete scribbleMap;
	scribbleMap = NULL;
    }
    return 0;
}   

#endif