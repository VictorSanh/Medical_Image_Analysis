#ifndef _dice_C
#define _dice_C

#include "CImg.h"
#include <iostream>
#include <list>

void diceScore(const CImg<float>& scribbleMap, const CImg<float>& groundTruth, std::list<float>& scores)
{
    int nbRegions = groundTruth.max();
    
    int countGroundTruth[1+nbRegions] = {0};
    int countScribbleMap[1+nbRegions] = {0};
    
    int intersection[1+nbRegions] = {0};

    for (int r = 0; r < scribbleMap.height(); r++){
        for (int c = 0; c < scribbleMap.width(); c++){
	  int groundTruthValue = int(groundTruth(r, c));
	  int scribbleMapValue = int(scribbleMap(r, c));
	  
	  countGroundTruth[groundTruthValue] = countGroundTruth[groundTruthValue] +1;
	  countScribbleMap[scribbleMapValue] = countScribbleMap[scribbleMapValue] +1;
	  
	  if (groundTruthValue == scribbleMapValue)
	    intersection[scribbleMapValue] = 1 + intersection[scribbleMapValue];
	}
    }  
    
    int i = 0;
    for (std::list<float>::iterator it=scores.begin();it!=scores.end(); it++){
	*it = float(2*intersection[i])/float(countGroundTruth[i] + countScribbleMap[i]);
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