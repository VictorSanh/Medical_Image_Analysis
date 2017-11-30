/*********************************************************
 * Read and treat images files in txt file to CImg files *
 ********************************************************/


#ifndef _segmentationManipulation_C
#define _segmentationManipulation_C

#include "CImg.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <list>
#include <vector>

using namespace std;
using namespace cimg_library;


vector< vector<int> > transpose(const vector< vector<int> > &data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    vector<vector<int> > result(data[0].size(),
	    vector<int>(data.size()));
    for (vector<int>::size_type i = 0; i < data[0].size(); i++)
	    for (vector<int>::size_type j = 0; j < data.size(); j++) {
		    result[i][j] = data[j][i];
	    }
    return result;
}

vector< vector<int> > load_txt(string name_file) {
    // Load a txt file containing integers and return a vecto<vector<int>> matrix
    ifstream infile(name_file.c_str());
    vector< vector<int> > data;
    string line;
    while (getline(infile, line)) { // read line by line
	    int n;
	    vector<int> vector_line;
	    istringstream iss(line);
	    while (iss >> n) { // read element by element in a line
		    vector_line.push_back(n);
	    }
	    data.push_back(vector_line);
    }
    return transpose(data);
    //return data;
}

CImg<int> matrix_to_cimg(const vector< vector<int> > &matrix) {
    // Transform a matrix to a CImg variable
    const int size_x = matrix.size();
    const int size_y = matrix[0].size();
    CImg<int> image(size_x, size_y, 1, 1, 0);
    for (int i = 0; i < size_x; i++) {
	    for (int j = 0; j < size_y; j++) {
		    image(i, j) = matrix[i][j];
	    }
    }
    return image;
}

CImg<int> load_txt_to_cimg(string name_file) {
    vector< vector<int> > data = load_txt(name_file.c_str());
    //cout << "Load Txt File into CImg. Size : " << data.size() << " " << data[0].size() << endl;
    CImg<int> image = matrix_to_cimg(data);
    return image;
}


#endif
