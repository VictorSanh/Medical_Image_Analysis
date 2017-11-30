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

#ifndef PARAMSH
#define PARAMSH

#include "CImg.h"
#include<stdio.h>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cimg_library;
using namespace std;

template <class T>
class CParams
{
  public:
    bool debugOutput;
    int brushSize;
    string scribbleFile;
    int nRegions;
    string imageFile;
    int numSteps;
    float smoothnessWeight; 
    float colorVariance;  
    float scribbleDistanceFactor; 
    string optimizationMethod; 
    string resultsFolder;
    string intputFolder;
    string groundTruthTxt;
    int outputEveryNSteps;
    float brushDensity;

    map<string, string> params;
    
    CParams()
    {
    };

    void readParams(string pfile)
    {
		//read parameters
		ifstream file(pfile.c_str());

		string s1, s2;
		char ab[256];
		while(!file.eof())
		{
			file >> ab;
			if(ab[0] == '#')
			{
        		file.ignore(10000, '\n');
        		continue;
			}
			s1 = ab;
			file >> ab;
			s2 = ab;
			params.insert(pair<string,string>(s1, s2));
		}
		file.close();

		resultsFolder = params["resultsfolder"];
		intputFolder = params["inputfolder"];
		scribbleFile = params["scribblefile"];
		imageFile = params["imagefile"];
		groundTruthTxt = params["groundTruthTxt"];

		if(params["debugoutput"] == "true") debugOutput = true;
		else debugOutput = false;

		stringstream s(params["brushsize"]);
		s >> brushSize;
		if(brushSize < 0)
		{
			cout << "WARNING: brushsize parameter is negative, setting to 1" << endl;
			brushSize = 1;
		}
		s.clear();

		s.str(params["outputeverynsteps"]);
		s >> outputEveryNSteps;
		if(outputEveryNSteps < 0)
		{
			cout << "WARNING: outputeverynsteps parameter is negative, setting to 100" << endl;
			outputEveryNSteps = 100;
		}
		s.clear();

		s.str(params["numsteps"]);
		s >> numSteps;
		if(numSteps < 0)
		{
			cout << "WARNING: numsteps parameter is negative, setting to 200" << endl;
			numSteps = 200;
		}
		s.clear();

		s.str(params["smoothnessweight"]);
		s >> smoothnessWeight;
		if(smoothnessWeight < 0)
		{
			cout << "WARNING: smoothnessweight parameter is negative, setting to 100" << endl;
			smoothnessWeight = 100;
		}
		s.clear();

		s.str(params["colorvariance"]);
		s >> colorVariance;
		if(colorVariance < 0)
		{
			cout << "WARNING: colorvariance parameter is negative, setting to 10" << endl;
			colorVariance = 10;
		}
		s.clear();

		s.str(params["scribbledistancefactor"]);
		s >> scribbleDistanceFactor;
		if(scribbleDistanceFactor < 0.4)
		{
			cout << "WARNING: scribbledistancefactor parameter is below 5 which can lead to artefacts, setting to 5" << endl;
			scribbleDistanceFactor = 5;
		}
		s.clear();

		s.str(params["optimizationmethod"]);
		s >> optimizationMethod;
		if(optimizationMethod != "zach" && optimizationMethod != "chambolle")
		{
			cout << "WARNING: optimizationmethod not supported, must be set to zach or chambolle, setting to zach" << endl;
			optimizationMethod = "zach";
		}
		s.clear();

		s.str(params["brushdensity"]);
		s >> brushDensity;
		if(brushDensity <= 0 || brushDensity > 1)
		{
			cout << "WARNING: brushdensity parameter must be in the range ]0,1], setting to 0.5" << endl;
			brushDensity = 0.5;
		}
		s.clear();
	}
};



#endif
