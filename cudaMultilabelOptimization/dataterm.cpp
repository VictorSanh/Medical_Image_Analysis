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





#ifndef _Dataterm_C
#define _Dataterm_C

#include "CImg.h"
#include <cassert>
#include "cudaDataterm.cuh"
#include <vector>
#include "cutil.h"
#include <iostream>

using namespace std;
using namespace cimg_library;

//for images in the range [0,256] setting values below 5 leads to artefacts in the segmentation due to spatial variances too close to 0
#define MINSCRIBBLEDISTANCEFACTOR 0.4

struct scribble
{
    int x, y; //scribble location
    float r, g, b; //scribble color
};


/***********************************
This class together with the cudaDataterm.cu file implements the paper
C. Nieuwenhuis and D. Cremers, Spatially Varying Color Distributions for Interactive Multi-Label Segmentation, Transactions on Pattern Analysis and Machine Intelligence, 2013
***********************************/

template <class T>
class Dataterm
{
  public:
    vector< vector< scribble > > scribbles;
    CImg<T> dataEnergy;
    int nRegions;

    Dataterm()
	{
	    this->nRegions = 0;
	}
  
	virtual ~Dataterm(){}

    virtual void computeDataEnergy(CImg<float> *img) = 0;

    virtual void addScribble(scribble &s, int region)
	{
	    if(region >= this->nRegions) //add new region label
	    {
			this->nRegions = region + 1;
			this->scribbles.resize(nRegions);
	    }
	    this->scribbles[region].push_back(s);
	}

    virtual void readScribblesFromMap(CImg<T> *scribbleMap, CImg<T> *img)
	{
	    scribble s;

	    cimg_forXY(*(scribbleMap), x, y)
	    {
			int region = (*scribbleMap)(x, y);
			if(region >= 0)
			{
				s.x = x;
				s.y = y;
				s.r = (*img)(x, y, 0, 0);
				s.g = (*img)(x, y, 0, 1);
				s.b = (*img)(x, y, 0, 2);
				addScribble(s, region);
			}
	    }
	    
	}
};

template <class T>
class SpatiallyVaryingParzenDataterm : public Dataterm<T>
{
  public:
    float colorVariance, scribbleDistanceFactor;

    SpatiallyVaryingParzenDataterm(float colorVariance, float scribbleDistanceFactor) : Dataterm<T> ()
	{
	    this->colorVariance = colorVariance;
	    this->scribbleDistanceFactor = scribbleDistanceFactor;
		if(this->scribbleDistanceFactor < MINSCRIBBLEDISTANCEFACTOR)
		{
			cout << "WARNING: scribbleDistanceFactor too small, setting to " << MINSCRIBBLEDISTANCEFACTOR << endl;
			this->scribbleDistanceFactor = MINSCRIBBLEDISTANCEFACTOR;
		}
	}

	~SpatiallyVaryingParzenDataterm()
	{
	}


	
	virtual void computeDataEnergy(CImg<float> *img)
	{
	    this->dataEnergy.assign(img->width(), img->height(), 1, this->nRegions);
	    int sumScribbles = 0;

		for(unsigned int n = 0; n < this->scribbles.size(); n++)
	    {
			sumScribbles += this->scribbles[n].size();
	    }
		if(sumScribbles == 0)
		{
			cout << "WARNING: no scribbles added to the data term class so far, use function readScribblesFromMap to read scribble information" << endl;
		}

		//Gaussian spatial variances
	    CImg<float> rhos(img->width(), img->height(), 1, this->nRegions);

	    float *scribbleData = new float[sumScribbles * sizeof(scribble)];
	    int *numScribbles = new int[this->scribbles.size()];

		//read scribbles from double vector to float array for cuda handling
	    int k = 0;
	    for(unsigned int i = 0; i < this->scribbles.size(); i++)
	    {
			numScribbles[i] = this->scribbles[i].size();
			for(unsigned int j = 0; j < this->scribbles[i].size(); j++)
			{
				scribbleData[k++] = this->scribbles[i][j].x;
				scribbleData[k++] = this->scribbles[i][j].y;
				scribbleData[k++] = this->scribbles[i][j].r;
				scribbleData[k++] = this->scribbles[i][j].g;
				scribbleData[k++] = this->scribbles[i][j].b;
			}
	    }

		parzenDataterm(this->dataEnergy.data(), img->data(), rhos.data(), scribbleData, numScribbles, sumScribbles, this->colorVariance, this->scribbleDistanceFactor, img->width(), img->height(), this->nRegions);

	    delete scribbleData;
	}

};




#endif
