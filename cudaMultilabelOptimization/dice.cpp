#ifndef _dice_C
#define _dice_C

#include "CImg.h"
#include <iostream>
#include <list>

void diceScore(const CImg<float>& estimatedSegmentation, const CImg<float>& groundTruth, std::list<float>& scores)
{
    int nbRegions = groundTruth.max();
    /*std::cout << "Max Groundtruth : " << nbRegions << " Min GroundTruth : " << groundTruth.min() << std::endl;
    std::cout << "Max Estimated : " << estimatedSegmentation.max() << " Min Estimated : " << estimatedSegmentation.min() << std::endl;
    std::cout <<groundTruth(0, 0) << " " <<estimatedSegmentation(0,0) <<std::endl;*/
    
    int countGroundTruth[1+nbRegions] = {0};
    int countEstimatedSegmentation[1+nbRegions] = {0};
    
    int intersection[1+nbRegions] = {0};

    for (int r = 0; r < estimatedSegmentation.height(); r++){
        for (int c = 0; c < estimatedSegmentation.width(); c++){
	  int groundTruthValue = int(groundTruth(r, c));
	  int scribbleMapValue = int(estimatedSegmentation(r, c));
	  
	  countGroundTruth[groundTruthValue] = countGroundTruth[groundTruthValue] +1;
	  countEstimatedSegmentation[scribbleMapValue] = countEstimatedSegmentation[scribbleMapValue] +1;
	  
	  if (groundTruthValue == scribbleMapValue)
	    intersection[scribbleMapValue] = 1 + intersection[scribbleMapValue];
	}
    }  
    
    int i = 0;
    for (std::list<float>::iterator it=scores.begin();it!=scores.end(); it++){
	*it = float(2*intersection[i])/float(countGroundTruth[i] + countEstimatedSegmentation[i]);
	i = i + 1;
    }
}


float averageDiceScore(std::list<float> scores)
{
    float average = 0;
    for (std::list<float>::iterator it=scores.begin();it!=scores.end(); it++)
	average = average + *it;
    average = average/float(scores.size());
    return average;
}

#endif