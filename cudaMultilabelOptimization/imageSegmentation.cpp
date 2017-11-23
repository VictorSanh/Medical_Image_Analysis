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


#ifndef _imageSegmentation_C
#define _imageSegmentation_C

#include "CImg.h"
#include "dataterm.cpp"
#include <time.h>
#include "cudaOptimization.cuh"
#include <iostream>


using namespace std;
using namespace cimg_library;

/***********************************
This class together with the cudaOptimization.cu file implements the paper
C. Nieuwenhuis and E. Toeppe and D. Cremers, A Survey and Comparison of Discrete and Continuous Multilabel Approaches for the Potts Model, International Journal of Computer Vision, 2013
***********************************/

template <class T>
class ImageSegmentation
{
public:

    int nRegions; //number of labels
    CImg<T> u, segmentation; //indicator function and final solution 
    CImg<T> scribbleImg; //image with scribbles
    CImgDisplay dispOriginal, dispSegmentation, dispData; //images used to display original image, segmentation result and data term
    vector< vector<T> > colors, colorsGrad; //set of colors for each label and the boundary of the label 
     

    ImageSegmentation(bool debugOutput)
	{
	    this->nRegions = 2;

	    cudaInit(debugOutput);

	    srand (time(NULL));

	}

    ~ImageSegmentation()
    {
    }

	void cudaInit(bool debugOutput = true)
	{
	    // Check available devices
	    int deviceCount;
	    cudaGetDeviceCount(&deviceCount);
	    if(debugOutput)
	    {
			fprintf(stderr,"Found %d GPU Devices",deviceCount);
			if ( deviceCount == 0 ) 
			{
				cout << "No CUDA devices found." << endl;
			}
	    }
	    int device;
	    cudaGetDevice( &device );
	    cudaDeviceProp deviceProp;
	    cudaGetDeviceProperties(&deviceProp, device);
	    if(debugOutput)
		cout << "Using CUDA device " << deviceProp.name << endl;
	}

//execute segmentation with user interaction

//dataterm: instance of dataterm class to compute data energy
//scribbleMap: return parameter indicating user scribble map containing the following values: unscribbled pixels -> -1, for specific label number -> set label number between 0 and MAXNREGIONS (defined in cudaOptimization.cu)
//img: original image
//brushSize: brush size of the scribles
//smoothnessWeight: weighting factor of regularization term
//debugOutput: output defug information or mute
//optimizationMethod: "zach" uses optimization method by Zach et al., "chambolle" uses optimization method by Chambolle et al.
//resultsFolder: path to folder where results are saved
//numSteps: number of steps to carry out optimization
//brushDensity: ratio ]0,..,1] of pixels in the scribble, which are actually used to calculate the data term - set to 1 for using all scribble points, set to lower values to save runtime and memory
//outputEveryNSteps: debug information is written to the console every outputEveryNSteps steps

    void executeInteractive(Dataterm<T> *dataterm, CImg<T> *scribbleMap, CImg<T> *img, int brushSize, float smoothnessWeight, bool debugOutput, string optimizationMethod, string resultsFolder, string outputName, int numSteps, float brushDensity = 1, int outputEveryNSteps = 100)
	{

	    vector< CImg<T> > newScribbles;
	    dispOriginal.assign(*img);

	    int regionToChange = -1;
		
	    bool update;
	    CImg<int> pos(1, 2), posd(1, 2);
	    int w = brushSize/2;

	    cout << "Please indicate regions: 1. draw scribble with mouse, 2. assign region label by pressing digit between 0 and 9" << endl;

	    while(!dispOriginal.button());

	    while(true)
	    {
			newScribbles.clear();
			update = false;
			
			//while mouse is clicked and dragged over the image add the points to the scribble and mark the scribble in the scribble map
			while(dispOriginal.button())
			{
				pos(0, 0) = dispOriginal.mouse_x();
				pos(0, 1) = dispOriginal.mouse_y();
				for(int x = -w; x <= w; x++)
				{
					for(int y = -w; y <= w; y++)
					{
						posd(0, 0) = pos(0, 0) + x;
						posd(0, 1) = pos(0, 1) + y;

						if(posd(0, 0) >= 0 && posd(0, 0) < img->width() && posd(0, 1) >= 0 && posd(0, 1) < img->height())
						{
							if((*(scribbleMap))(posd(0, 0), posd(0, 1)) == -1)
							{
								if(rand() / (float)RAND_MAX < brushDensity)
								{
									newScribbles.push_back(posd);
									update = true;
								}
								(*(scribbleMap))(posd(0, 0), posd(0, 1)) = -2; //mark to avoid adding again
							}
						}
					}
				}
			}
			//assign the drawn scribble to a region label between 0 and 9
			if(update)
			{
				regionToChange = -1;
				while(regionToChange == -1)
				{
					if(dispOriginal.is_key0())
					{
						regionToChange = 0;
					}
					else if(dispOriginal.is_key1())
					{
						regionToChange = 1;
					}
					else if(dispOriginal.is_key2())
					{
						regionToChange = 2;
					}
					else if(dispOriginal.is_key3())
					{
						regionToChange = 3;
					}
					else if(dispOriginal.is_key4())
					{
						regionToChange = 4;
					}
					else if(dispOriginal.is_key5())
					{
						regionToChange = 5;
					}
					else if(dispOriginal.is_key6())
					{
						regionToChange = 6;
					}
					else if(dispOriginal.is_key7())
					{
						regionToChange = 7;
					}
					else if(dispOriginal.is_key8())
					{
						regionToChange = 8;
					}
					else if(dispOriginal.is_key9())
					{
						regionToChange = 9;
					}
				}
				//check if new region label must be added
				if(regionToChange >= this->nRegions)
				{
					this->nRegions = regionToChange + 1;
				}
				for(unsigned int i = 0; i < newScribbles.size(); i++)
				{
					(*(scribbleMap))(newScribbles[i](0,0), newScribbles[i](0,1)) = regionToChange; 
				}

				//read scribble information (location and color) from scribbleMap into the dataterm scribble data structure
				dataterm->readScribblesFromMap(scribbleMap, img);
				//compute data term
				dataterm->computeDataEnergy(img);
			    
				//execute optimization
				executeAutomatic(&dataterm->dataEnergy, img, smoothnessWeight, debugOutput, optimizationMethod, numSteps, outputEveryNSteps);
				drawAndSaveResults(img, scribbleMap, &dataterm->dataEnergy, resultsFolder, outputName);
				sleep(2); //required to avoid occasional display problems with CImg library
				dispOriginal.assign(scribbleImg);
				dispSegmentation.assign(this->segmentation);
			}
	    }
	}



//execute segmentation based on loaded data energy

//dataEnergy: return parameter defining data term for each pixel (x,y) for each label (in each channel of the image)
//img: original image
//smoothnessWeight: weighting factor of regularization term
//debugOutput: output debug information or mute
//optimizationMethod: "zach" uses optimization method by Zach et al., "chambolle" uses optimization method by Chambolle et al.
//numSteps: number of steps to carry out optimization
//outputEveryNSteps: debug information is written to the console every outputEveryNSteps steps

    
    void executeAutomatic(CImg<T> *dataEnergy, CImg<T> *img, float smoothnessWeight, bool debugOutput, string optimizationMethod, int numSteps, int outputEveryNSteps = 100)
	{
	    this->nRegions = dataEnergy->spectrum();

	    u.assign(img->width(), img->height(), 1, this->nRegions);
	
		//compute TV weighting function g(x) to make segmentation boundary align with image edges
	    CImg<T> edge(img->width(), img->height(), 1, 1, 1);
	    CImgList<T> grad(2);

	    grad = img->get_gradient();
	    grad[0] /= grad[0].max();
	    grad[1] /= grad[1].max();

//compute factor for edge indicator function g, can be estimated from image, we found 5 a better value	    
/*	    float edgeVar = 0;
	    float n;
	    cimg_forXY(grad[0], x, y)
	    {
		n = 0;
		cimg_forC(grad[0], c)
		{
		    n += grad[0](x, y, 0, c) * grad[0](x, y, 0, c) + grad[1](x, y, 0, c) * grad[1](x, y, 0, c);
		}
		edgeVar += n;
	    }
	    edgeVar /= img->width() * img->height();

	    edgeVar = 1/(2 * edgeVar);

	    cout << "edgeVar " << edgeVar << endl;
*/

//compute edge indicator function g(x)
  	   float  edgeVar = 5;
	   
	    cimg_forXY(grad[0], x, y)
	    {
			edge(x, y) = 0;
			cimg_forC(grad[0], c)
			{
				edge(x, y) += grad[0](x, y, 0, c) * grad[0](x, y, 0, c);
				edge(x, y) += grad[1](x, y, 0, c) * grad[1](x, y, 0, c);
			}
			edge(x, y) = exp(-edgeVar * sqrt(edge(x, y)));
	    }

	    clock_t tt1, tt2;

	    if(optimizationMethod == "zach")
	    {
			if(debugOutput) cout << "Using optimization method by Zach et al." << endl;
			tt1 =clock();
			//The following function is based on the paper
			//C. Zach, D. Gallup, J. Frahm and M. Niethammer, Fast global labeling for realtime stereo using multiple plane sweeps, Vision, Modeling and Visualization Workshop (VMV), 2008
			//and the formulation used in formula (8) in the paper
			//C. Nieuwenhuis and E. Toeppe and D. Cremers, A Survey and Comparison of Discrete and Continuous Multilabel Approaches for the Potts Model, International Journal of Computer Vision, 2013
			zachPrimalDual(u.data(), edge.data(), dataEnergy->data(), smoothnessWeight, img->width(), img->height(), this->nRegions, numSteps, outputEveryNSteps, debugOutput);
			tt2 =clock();
	    }
	    else if(optimizationMethod == "chambolle")
	    {
			if(debugOutput) cout << "Using optimization method by Chambolle et al." << endl;
			tt1 =clock();
			//The following function is based on the paper
			//A. Chambolle, D. Cremers, T. Pock, A Convex Approach to Minimal Partitions, SIAM Journal on Imaging Sciences, 2012
			//and its formulation used in formula (11) of the paper
			//C. Nieuwenhuis and E. Toeppe and D. Cremers, A Survey and Comparison of Discrete and Continuous Multilabel Approaches for the Potts Model, International Journal of Computer Vision, 2013
			chambollePrimalDual(u.data(), edge.data(), dataEnergy->data(), smoothnessWeight, img->width(), img->height(), this->nRegions, numSteps, outputEveryNSteps, debugOutput);
			tt2 =clock();
	    }
	    else return;
	    if(debugOutput) cout << "Runtime: " << double(tt2-tt1)/CLOCKS_PER_SEC << endl;

	    binarizeResult(u);

	}

	//obtain binary solution from continuous relaxed result by assigning each pixel to maximum region label
    void binarizeResult(CImg<T> &u)
	{
	    cimg_forXY(u, x, y)
	    {
			float uMax = 0;
			int uMaxIndex = 0;
			for(int i = 0; i < this->nRegions; i++)
			{
				if(u(x, y, 0, i) > uMax)
				{
					uMax = u(x, y, 0, i);
					uMaxIndex = i;
				}
			}
			for(int i = 0; i < this->nRegions; i++)
			{
				u(x, y, 0, i) = 0;
			}
			u(x, y, 0, uMaxIndex) = 1;
	    }	
	}

    void drawSegmentation(CImg<T> *img)
	{
	    setColors();

	    segmentation.assign(img->width(), img->height(), 1, 3, 0);
	    segmentation.assign(*img);

	    bool grad;
	    cimg_forXY(segmentation, x, y)
	    {
			for(int i = 0; i < u.spectrum(); i++)
			{
				if(u(x, y, 0, i) == 1)
				{
					segmentation(x, y, 0, 0) = colors[i][0];
					segmentation(x, y, 0, 1) = colors[i][1];
					segmentation(x, y, 0, 2) = colors[i][2];
				}

				grad = false;
				if(x > 0 &&  x < img->width() - 1 && y > 0 && y < img->height() - 1)
				{
					if(u(x, y, 0, i) - u(x + 1, y, 0, i) > 0.5 || u(x, y, 0, i) - u(x, y + 1, 0, i) > 0.5 ||
					   u(x, y, 0, i) - u(x - 1, y, 0, i) > 0.5 || u(x, y, 0, i) - u(x, y - 1, 0, i) > 0.5)
						grad = true;
				}
				else grad = true;
					
				if(grad)
				{
					for(int a = -1; a <= 1; a++)
					{
						if(x + a >= 0 && x + a < img->width())
						{
							for(int b = -1; b <= 1; b++)
							{
								if(y + b >= 0 && y + b < img->height())
								{
									if(u(x + a, y + b, 0, i) == 1)
									{
										segmentation(x, y, 0, 0) = colorsGrad[i][0];
										segmentation(x, y, 0, 1) = colorsGrad[i][1];
										segmentation(x, y, 0, 2) = colorsGrad[i][2];
									}
								}
							}
						}
					}
				}
			}
	    }
	    segmentation += *img;
	    segmentation = (segmentation - segmentation.min())/(segmentation.max() - segmentation.min()) * 255;
	}

    void drawAndSaveResults(CImg<T> *img, CImg<T> *scribbleMap, CImg<T> *dataEnergy, string resultsFolder, string outputName)
	{

	    drawScribbles(img, scribbleMap);
	    drawSegmentation(img);

	    CImg<T> normImg;
	    
	    cout <<"Saving " <<outputName.c_str() <<endl;
	    
	    char t[255];
	    string fileName = resultsFolder + "scribbleMap/";
	    //sprintf(t, "%sscribbleMap.cimg", fileName.c_str());
	    sprintf(t, "%s%s.cimg", fileName.c_str(), outputName.c_str());
	    scribbleMap->save_cimg(t);

	    fileName = resultsFolder + "scribbleImg/";
	    sprintf(t, "%s%s.bmp", fileName.c_str(), outputName.c_str());
	    scribbleImg.save_bmp(t);

	    fileName = resultsFolder + "segmentationMap/";
	    sprintf(t, "%s%s.cimg", fileName.c_str(), outputName.c_str());
	    segmentation.save_cimg(t);

	    fileName = resultsFolder + "dataEnergy/";
	    sprintf(t, "%s%s.cimg", fileName.c_str(), outputName.c_str());
	    dataEnergy->save_cimg(t);

	    fileName = resultsFolder + "segmentationImg/";
	    sprintf(t, "%s%s.bmp", fileName.c_str(), outputName.c_str());
	    normImg = segmentation.get_normalize(0,255);
	    normImg.save_bmp(t);

	    fileName = resultsFolder + "u/";
	    sprintf(t, "%s%s.cimg", fileName.c_str(), outputName.c_str());
	    u.save_cimg(t);
	}

    void drawScribbles(CImg<T> *img, CImg<T> *scribbleMap)
	{
	    setColors();
	    scribbleImg.assign(*img);
	    int region;
	    cimg_forXY(*scribbleMap, x, y)
	    {
			region = (*scribbleMap)(x,y);
			if(region >= 0)
			{
				this->scribbleImg(x, y, 0, 0) = colors[region][0];
				this->scribbleImg(x, y, 0, 1) = colors[region][1];
				this->scribbleImg(x, y, 0, 2) = colors[region][2];
			}
	    }
	}

    void setColors()
	{

	    colors.resize(this->nRegions);
	    colorsGrad.resize(this->nRegions);
	    for(int i = 0; i < this->nRegions; i++)
		{
		    colors[i].resize(3);
		    colorsGrad[i].resize(3);

		    if(i % 10 == 0)
		    {
				colors[i][0] = 250;		
				colors[i][1] = 0;		
				colors[i][2] = 0;		

				colorsGrad[i][0] = 255;		
				colorsGrad[i][1] = 0;		
				colorsGrad[i][2] = 0;		
		    }
		    else if(i % 10 == 1)
		    {
				colors[i][0] = 0;		
				colors[i][1] = 250;		
				colors[i][2] = 0;		

				colorsGrad[i][0] = 0;		
				colorsGrad[i][1] = 255;		
				colorsGrad[i][2] = 0;		
		    }
		    else if(i % 10 == 2)
		    {
				colors[i][0] = 0;		
				colors[i][1] = 0;		
				colors[i][2] = 250;		

				colorsGrad[i][0] = 0;		
				colorsGrad[i][1] = 0;		
				colorsGrad[i][2] = 255;		
		    }
		    else if(i % 10 == 3)
		    {
				colors[i][0] = 250;		
				colors[i][1] = 250;		
				colors[i][2] = 0;		

				colorsGrad[i][0] = 255;		
				colorsGrad[i][1] = 255;		
				colorsGrad[i][2] = 0;		
		    }
		    else if(i % 10 == 4)
		    {
				colors[i][0] = 250;		
				colors[i][1] = 0;		
				colors[i][2] = 250;		

				colorsGrad[i][0] = 255;		
				colorsGrad[i][1] = 0;		
				colorsGrad[i][2] = 255;		
		    }
		    else if(i % 10 == 5)
		    {
				colors[i][0] = 0;		
				colors[i][1] = 250;		
				colors[i][2] = 250;		

				colorsGrad[i][0] = 0;		
				colorsGrad[i][1] = 255;		
				colorsGrad[i][2] = 255;		
		    }
		    else if(i % 10 == 6)
		    {
				colors[i][0] = 125;		
				colors[i][1] = 250;		
				colors[i][2] = 0;		

				colorsGrad[i][0] = 130;		
				colorsGrad[i][1] = 255;		
				colorsGrad[i][2] = 0;		
		    }
		    else if(i % 10 == 7)
		    {
				colors[i][0] = 125;		
				colors[i][1] = 0;		
				colors[i][2] = 250;		

				colorsGrad[i][0] = 130;		
				colorsGrad[i][1] = 0;		
				colorsGrad[i][2] = 255;		
		    }
		    else if(i % 10 == 8)
		    {
				colors[i][0] = 0;		
				colors[i][1] = 125;		
				colors[i][2] = 250;		

				colorsGrad[i][0] = 0;		
				colorsGrad[i][1] = 130;		
				colorsGrad[i][2] = 255;		
		    }
		    else if(i % 10 == 9)
		    {
				colors[i][0] = 250;		
				colors[i][1] = 125;		
				colors[i][2] = 0;		

				colorsGrad[i][0] = 255;		
				colorsGrad[i][1] = 130;		
				colorsGrad[i][2] = 0;		
		    }
		}
    }
};

#endif
