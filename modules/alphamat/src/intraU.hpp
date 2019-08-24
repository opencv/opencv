// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "Eigen/Sparse"
using namespace Eigen;
using namespace nanoflann;
using namespace std;
using namespace cv;


typedef vector<vector<double>> my_vector_of_vectors_t;
typedef vector<set<int, greater<int>>> my_vector_of_set_t;
vector<int> orig_ind; 

void generateFVectorIntraU(my_vector_of_vectors_t &samples, Mat &img, Mat& tmap)
{
	// CV_Assert(img.depth() == CV_8U);

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;

	
	int unk_count = 0; 
	int i,j,k;
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			float pix = tmap.at<uchar>(i,j);
    		if(pix == 128)
    			unk_count++;
		}
	samples.resize(unk_count);
	orig_ind.resize(unk_count);
	
	int c1 = 0;	
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			float pix = tmap.at<uchar>(i,j);
    		if(pix == 128){							//collection of unknown pixels samples
				samples[c1].resize(dim);		
				samples[c1][0] = img.at<cv::Vec3b>(i,j)[0];
				samples[c1][1] = img.at<cv::Vec3b>(i,j)[1];
				samples[c1][2] = img.at<cv::Vec3b>(i,j)[2];
				samples[c1][3] = (double(i)/nRows)/20;
				samples[c1][4] = (double(j)/nCols)/20;
				orig_ind[c1] = i*nCols+j;
    			c1++;
    		}
		}

	// cout << "feature vectors done"<<endl;
}



void kdtree_intraU(Mat &img, Mat& tmap, my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples)
{
	const double max_range = 20;

	// Generate feature vectors for intra U:
	generateFVectorIntraU(samples, img, tmap);
	

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------

	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();
	// do a knn search with ku  = 5
	const size_t num_results = 5+1; 

	int i,j;
	int N = samples.size();		//no. of unknown samples

	// just for testing purpose ...delete this later!
	int c = 0; 

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);



	indm.resize(N);
	inds.resize(N);
	for(i = 0; i < N; i++){
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &samples[i][0], nanoflann::SearchParams(10));	

		// cout << "knnSearch(nn="<<num_results<<"): \n";
		indm[i].resize(num_results-1);
		for (j = 1; j < num_results; j++){
			// cout << "ret_index["<<j<<"]=" << ret_indexes[j] << " out_dist_sqr=" << out_dists_sqr[j] << endl;
			inds[i].insert(ret_indexes[j]);
			indm[i][j-1] = ret_indexes[j];
		}
		c++;
		// if(c == 5)
		// 	return;
	}
}

double l1norm(vector<double>& x, vector<double>& y){
	double sum = 0;
	for(int i = 0; i < dim; i++)
		sum += abs(x[i]-y[i]);
	return sum;
}

void intraU(my_vector_of_vectors_t& indm, my_vector_of_set_t& inds, my_vector_of_vectors_t& samples, my_vector_of_vectors_t& Euu, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu){

	// input: indm, inds, samples
	int num_nbr = 5;
	int n = indm.size();  //num of unknown samples
	

	int i,j, curr_ind, nbr_ind;
	int count = 0;
	for(i = 0; i < n; i++){
		for(j = 0; j < num_nbr; j++){
			// lookup curr_ind in indm[i][j](jth nbr entry)
			curr_ind = i;
			nbr_ind = indm[i][j];
			if(inds[nbr_ind].find(curr_ind) == inds[nbr_ind].end()){
				indm[nbr_ind].push_back(curr_ind);
				count++;
			}
		}
	}

	
	my_vector_of_vectors_t weights;
	// SparseMatrix<double> Wuu(N, N), Duu(N, N);
	typedef Triplet<double> T;
    vector<T> triplets, td;
    triplets.reserve(num_nbr*n + count);
    td.reserve(num_nbr*n + count);

	double weight;
	for(i = 0; i < n; i++){
		// weights[i].resize(indm[i].size());
		for(j = 0; j < indm[i].size(); j++){
			// weights[i][j] = max(l1norm(samples[i], samples[j]), 0.0);
			weight = max(l1norm(samples[i], samples[j]), 0.0);
			nbr_ind = indm[i][j];
			triplets.push_back(T(orig_ind[i], orig_ind[nbr_ind], weight));
			td.push_back(T(orig_ind[i], orig_ind[i], weight));
		}
	}

	Wuu.setFromTriplets(triplets.begin(), triplets.end());
	Duu.setFromTriplets(td.begin(), td.end());
	// cout<<"intraU weights computed"<<endl;
	// return Wuu;
	// Euu	
}

void UU(Mat& image, Mat& tmap, SparseMatrix<double>& Wuu, SparseMatrix<double>& Duu){
	my_vector_of_vectors_t samples, indm, Euu;
	my_vector_of_set_t inds;
	// string img_path = "../../data/input_lowres/plasticbag.png";
	// image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

	// string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
 //    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);

	kdtree_intraU(image, tmap, indm, inds, samples);
	// cout<<"KD Tree done"<<endl;
	int N = image.rows * image.cols;
	intraU(indm, inds, samples, Euu, Wuu, Duu);
	cout<<"Intra U Done"<<endl;

}

/*

int main()
{
	Mat image,tmap;
	string img_path = "../../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

	string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
    // SparseMatrix<double> Wuu = UU(image, tmap);

}

*/
