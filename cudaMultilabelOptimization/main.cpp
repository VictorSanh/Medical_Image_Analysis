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
#include <fstream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
namespace bfs = boost::filesystem;



int computeSegmentationFromParameterFile()
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
	if (params.scribbleFile.find(".txt") != std::string::npos)
	    scribbleMap = new CImg<float>(load_txt_to_cimg(params.scribbleFile.c_str()));
	else
	    scribbleMap = new CImg<float>(params.scribbleFile.c_str());


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

    if(interactive){
	segmentation.executeInteractive(dataterm,
					scribbleMap,
					img,
					params.brushSize,
					params.smoothnessWeight,
					params.debugOutput,
					params.optimizationMethod,
					params.resultsFolder,
					outputName,
					params.numSteps,
					params.brushDensity,
					params.outputEveryNSteps);
    }
    else //use loaded scribble map
    {
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
	segmentation.segmentation.display();
    }


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

CImg<float> estimateSegmentation(CImg<float> *scribbleMap, CImg<float> *img, CParams<float> params, string outputName, bool save, bool display)
{
    //Normalize image to range [0,255]
    float imgMax = img->max();
    *img = *img / imgMax * 255;

    //initialize data term
    if(params.debugOutput) cout << "Creating dataterm" << endl;
    Dataterm<float> *dataterm = new SpatiallyVaryingParzenDataterm<float>(params.colorVariance, params.scribbleDistanceFactor);

    //initialize segmentation
    if(params.debugOutput) cout << "Initialize cuda for segmentation" << endl;
    ImageSegmentation<float> segmentation(params.debugOutput);


    if(scribbleMap->width() != img->width() || scribbleMap->height() != img->height())
    {
	cout << "WARNING: scribble file size does not match image size" << endl;
    }

    //read number of regions from scribble file
    int m = scribbleMap->max() - scribbleMap->min();
    params.nRegions = m + 1;

    int  edgeVar = 5;
    char outputParameters[255];
    sprintf(outputParameters, "%0.1f_%0.3f_%i_%0.1f", params.colorVariance, params.smoothnessWeight, edgeVar, params.scribbleDistanceFactor);
    string outputNameWithParameters = outputName + outputParameters;

    //execute segmentation algorithm
    dataterm->readScribblesFromMap(scribbleMap, img);
    dataterm->computeDataEnergy(img);
    segmentation.executeAutomatic(&(dataterm->dataEnergy),
				  img,
				  params.smoothnessWeight,
				  params.debugOutput,
				  params.optimizationMethod,
				  params.numSteps,
				  params.outputEveryNSteps);
    if(save)
    {
	segmentation.saveResults(img,
				 scribbleMap,
				 params.resultsFolder,
				 outputNameWithParameters);
    }
    if(display)
    {
	segmentation.displayScribbles(img, scribbleMap);
	cin.get();
    }


    //release memory
    if(dataterm)
    {
	delete dataterm;
	dataterm = NULL;
    }

    return segmentation.u;
}


int randomScribbleAnalysis(bool randomScribbleGeneration)
{
    //Read Parameters
    CParams<float> params;
    params.readParams("randomScribbleParameters.txt");
    cout << "Parameters read" << endl;
    cout <<"Image File : " <<params.imageFile.c_str() <<endl;
    cout <<"Input Folder : " <<params.intputFolder.c_str() <<endl;
    cout <<"OutputFolder : " <<params.resultsFolder.c_str() <<endl;
    cout <<"Ground Truth Txt : " <<params.groundTruthTxt.c_str() <<endl <<endl <<endl;


    //Computing and writing Random Scribbles
    cout << "Starting Bash Script" <<endl;
    string labelsFile = params.groundTruthTxt;
    string svgFileName = params.intputFolder;
    string call = "./randomScribbleGeneration " + labelsFile + " " + svgFileName;
    if (randomScribbleGeneration)
	system(call.c_str());
    cout << "Quitting Bash Script" <<endl <<endl <<endl;


    //read image and normalize to range [0,255]
    CImg<float> *img = new CImg<float>(params.imageFile.c_str());
    float imgMax = img->max();
    *img = *img / imgMax * 255;
    cout << "Image Loaded - Height: " << img->height() << " Width: " << img->width() <<endl;


    //Read ground truth
    string groundTruthFileName = params.intputFolder + "groundTruth.txt";
    CImg<float> *groundTruth = new CImg<float>(transpose(load_txt_to_cimg(groundTruthFileName.c_str())));
    cout << "Ground Truth Txt Map Loaded - Height: " << groundTruth->height() << " Width: " << groundTruth->width() <<endl <<endl <<endl;


    //Allocate Memory for Scribble Map
    CImg<float> *scribbleMap = NULL;


    //Iterate over all the Random Scribble Maps
    int k = 0;
    bool save = false;
    float avg = 0;
    string name = params.imageFile;
    name = name.substr(name.find_last_of("/")+1);
    name = name.substr(0, name.size()-4);


    string csvFileName = params.resultsFolder + "diceScores.csv";
    ofstream csvDump;
    csvDump.open(csvFileName.c_str());
    csvDump <<"ScribbleId" <<"," <<"AveraDiceScore" <<endl;


    bfs::path targetDir((params.intputFolder).c_str());
    bfs::directory_iterator it(targetDir), eod;
    BOOST_FOREACH(bfs::path const &p, std::make_pair(it, eod))
    {
	if(bfs::is_regular_file(p))
	    if (p.c_str()!= (params.intputFolder + "groundTruth.cimg") && p.c_str()!= (params.intputFolder + "groundTruth.txt") && p.c_str()!= (params.imageFile) && p.c_str() != params.groundTruthTxt)
	    {
		scribbleMap = new CImg<float>(load_txt_to_cimg(p.c_str()));
		cout << "Scribble Map Loaded - Height: " << scribbleMap->height() << " Width: " << scribbleMap->width() <<endl;

		string id = p.c_str();
		id = id.substr(id.find_last_of("/")+1);
		id = id.substr(0, id.size()-4);

		save = (k%params.outputEveryNSteps)==0;
		CImg<float> estimated = estimateSegmentation(scribbleMap, img, params, name + "_" + id + "_", save, false);

		std::list<float> scores(1+(*groundTruth).max());
		diceScore(estimated, (*groundTruth), scores);
		/*cout << "Dice Scores Details : "  <<endl;
		for (list<float>::const_iterator it = scores.begin(); it != scores.end(); ++it)
		    std::cout << *it << "\n";*/
		avg = averageDiceScore(scores);
		cout << "Average Dice Score: " << avg <<endl;
		k = k + 1;

		csvDump <<(params, name + "_" + id) <<"," <<avg <<endl;
	    }
    }
    csvDump.close();

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
    if(groundTruth)
    {
	delete groundTruth;
	groundTruth = NULL;
    }


    return 0;
}


int main()
{
    //int p = computeSegmentationFromParameterFile();
    randomScribbleAnalysis(true);
    //computeSegmentationFromParameterFile();
    return 0;
}

#endif
