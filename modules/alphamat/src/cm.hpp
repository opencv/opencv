// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifndef cm
// #define cm
//header file content


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


void generateFVectorCM(my_vector_of_vectors_t &samples, Mat &img)
{
	// CV_Assert(img.depth() == CV_8U);

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;

	samples.resize(nRows*nCols);
	
	int i,j,k;	
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			samples[i*nCols+j].resize(dim);		
			samples[i*nCols+j][0] = img.at<cv::Vec3b>(i,j)[0]/255.0;
			samples[i*nCols+j][1] = img.at<cv::Vec3b>(i,j)[1]/255.0;
			samples[i*nCols+j][2] = img.at<cv::Vec3b>(i,j)[2]/255.0;
			samples[i*nCols+j][3] = double(i)/nRows;
			samples[i*nCols+j][4] = double(j)/nCols;
		}

	// cout << "feature vectors done"<<endl;
}


void kdtree_CM(Mat &img, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, unordered_set<int>& unk)
{
	
	// Generate feature vectors for intra U:
	generateFVectorCM(samples, img);	

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();

	// do a knn search with cm = 20
	const size_t num_results = 20+1; 

	int N = unk.size();

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);

	indm.resize(N);
	int i = 0;
	for(unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++){
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &samples[*it][0], nanoflann::SearchParams(10));	

		// cout << "knnSearch(nn="<<num_results<<"): \n";
		indm[i].resize(num_results-1);
		for (int j = 1; j < num_results; j++){
			// cout << "ret_index["<<j<<"]=" << ret_indexes[j] << " out_dist_sqr=" << out_dists_sqr[j] << endl;
			indm[i][j-1] = ret_indexes[j];
		}
		i++;
	}
}



void lle(my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, float eps, unordered_set<int>& unk
						 , SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm){
	
	int k = indm[0].size(); //number of neighbours that we are considering 
	int n = indm.size(); //number of unknown pixels
	int N = samples.size();
	// SparseMatrix<double> Wcm(N,N), Dcm(N,N);
	typedef Triplet<double> T;
    vector<T> triplets, td;
    triplets.reserve(k*n);
    td.reserve(k*n);

	my_vector_of_vectors_t wcm;
	wcm.resize(n);
	
	Mat C(20, 20, DataType<float>::type), rhs(20, 1, DataType<float>::type), Z(3, 20, DataType<float>::type), weights(20, 1, DataType<float>::type);
	C = 0;
	rhs = 1;

	int i, ind = 0; 
	// cout<<n<<" "<<k<<endl;
	for(unordered_set<int>::iterator it = unk.begin(); it != unk.end(); it++){
		// filling values in Z
		i = *it;
		int index_nbr;
		for(int j = 0; j < k; j++){
			index_nbr = indm[ind][j];
			for(int p = 0; p < dim-2; p++)
				Z.at<float>(p,j) = samples[index_nbr][p] - samples[i][p];
		}
		

		// C1 = Z1.transpose()*Z1;
		// C1.diagonal().array() += eps;
		// weights1 = C1.ldlt().solve(rhs1);
		// weights1 /= weights1.sum();
		// cout<<weights1<<endl;
		// exit(0);

		
		C = Z.t()*Z;
		for(int p = 0; p < k; p++)
			C.at<float>(p,p) += eps;
		// cout<<"determinant: "<<determinant(C)<<endl;
		solve(C, rhs, weights, DECOMP_CHOLESKY);
		float sum = 0;

		for(int j = 0; j < k; j++)
			sum += weights.at<float>(j,0);;
		// cout<<"SUM:"<<sum<<endl;
		// exit(0);
		for(int j = 0; j < k; j++){
			weights.at<float>(j,0) /= sum;
			triplets.push_back(T(i, indm[ind][j], weights.at<float>(j,0)));
			// if(ind == 0){
			// 	cout<<i<<" "<<indm[ind][j]<<" "<<weights.at<float>(j,0)<<endl;
			// }
			td.push_back(T(i, i, weights.at<float>(j,0)));
		}
		// cout<<weights;
		// wcm[ind].resize(k);
		// for(int j = 0; j < k; j++){
		// 	wcm[ind][j] = weights.at<float>(j,0);

		// }
		
		ind++;
	}

    Wcm.setFromTriplets(triplets.begin(), triplets.end());
    Dcm.setFromTriplets(td.begin(), td.end());
    // return Wcm;
}

void cm(Mat& image, Mat& tmap, SparseMatrix<double>& Wcm, SparseMatrix<double>& Dcm){
	
	my_vector_of_vectors_t samples, indm, Euu;

    int i, j;
    unordered_set<int> unk;
    for(i = 0; i < tmap.rows; i++)
    	for(j = 0; j < tmap.cols; j++){
    		float pix = tmap.at<uchar>(i,j);
    		if(pix == 128)
    			unk.insert(i*tmap.cols+j);
    	}

    // cout<<"UNK: "<<unk.size()<<endl;

    kdtree_CM(image, indm, samples, unk);
	// cout<<"KD Tree done"<<endl;
	float eps = 0.001;
	lle(indm, samples, eps, unk, Wcm, Dcm);
	cout<<"cm DONE"<<endl;
	
}


/*
int main()
{
	Mat image,tmap;
	my_vector_of_vectors_t samples, indm, Euu;
	string img_path = "../../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
  	int N = image.rows*image.cols; 
  	SparseMatrix<double> Wcm(N,N), Dcm(N,N);
    cm(image, tmap, Wcm, Dcm);
}

*/
// #endif






































