// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifndef KtoU
// #define KtoU

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "Eigen/Sparse"
using namespace Eigen;
using namespace nanoflann;
using namespace std;
using namespace cv;
// const int dim = 5;

typedef vector<vector<double>> my_vector_of_vectors_t;
typedef vector<set<int, greater<int>>> my_vector_of_set_t;
map<pair<int, int>, int> ind; //original index mapping set - (unk, fg, bg)



void generateFVectorKtoU(my_vector_of_vectors_t& fv_unk, my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg,
						 Mat &img, Mat &tmap)
{
	// CV_Assert(img.depth() == CV_8U);
	int nRows = img.rows;
	int nCols = img.cols;

	int fg = 0, bg = 0, unk = 0, c1 = 0, c2 = 0, c3 = 0;
	int i,j,k;
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			double pix = tmap.at<uchar>(i,j);
    		if(pix == 128)
    			unk++;
    		else if(pix > 200)
    			fg++;
    		else bg++;
		}

	fv_fg.resize(fg);
	fv_bg.resize(bg);
	fv_unk.resize(unk);
		
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			double pix = tmap.at<uchar>(i,j);
    		if(pix == 128){
    			fv_unk[c1].resize(dim);		
				fv_unk[c1][0] = img.at<cv::Vec3b>(i,j)[0]/255.0;
				fv_unk[c1][1] = img.at<cv::Vec3b>(i,j)[1]/255.0;
				fv_unk[c1][2] = img.at<cv::Vec3b>(i,j)[2]/255.0;
				fv_unk[c1][3] = double(i)*10/nRows;
				fv_unk[c1][4] = double(j)*10/nCols;
				ind[{c1,0}] = i*nCols+j;
				c1++;
    		}
    		else if(pix > 200){
    			fv_fg[c2].resize(dim);		
				fv_fg[c2][0] = img.at<cv::Vec3b>(i,j)[0]/255.0;
				fv_fg[c2][1] = img.at<cv::Vec3b>(i,j)[1]/255.0;
				fv_fg[c2][2] = img.at<cv::Vec3b>(i,j)[2]/255.0;
				fv_fg[c2][3] = double(i)*10/nRows;
				fv_fg[c2][4] = double(j)*10/nCols;
				ind[{c2,1}] = i*nCols+j;
    			c2++;
    		}
    		else{
    			fv_bg[c3].resize(dim);	
				fv_bg[c3][0] = img.at<cv::Vec3b>(i,j)[0]/255.0;
				fv_bg[c3][1] = img.at<cv::Vec3b>(i,j)[1]/255.0;
				fv_bg[c3][2] = img.at<cv::Vec3b>(i,j)[2]/255.0;
				fv_bg[c3][3] = double(i)*10/nRows;
				fv_bg[c3][4] = double(j)*10/nCols;
				ind[{c3,2}] = i*nCols+j;
    			c3++;
    		}
		}

	// cout << "feature vectors done "<<c3<<endl;
}


void kdtree_KtoU(Mat &img, Mat &tmap, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& fv_unk, 
				 my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg)
{
	
	// Generate feature vectors for intra U:
	generateFVectorKtoU(fv_unk, fv_fg, fv_bg, img, tmap);	

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index_fg(dim /*dim*/, fv_fg, 10 /* max leaf */ );
	mat_index_fg.index->buildIndex();

	my_kd_tree_t mat_index_bg(dim /*dim*/, fv_bg, 10 /* max leaf */ );
	mat_index_bg.index->buildIndex();

	// do a knn search with cm = 20
	const size_t num_results = 7; 

	int N = fv_unk.size();

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);

	indm.resize(N);
	for(int i = 0; i < fv_unk.size(); i++){
		indm[i].resize(2*num_results);

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index_fg.index->findNeighbors(resultSet, &fv_unk[i][0], nanoflann::SearchParams(10));	
		for (int j = 0; j < num_results; j++){
			// cout << "$$$$$$$ret_index["<<j<<"]=" << ret_indexes[j] << " out_dist_sqr=" << out_dists_sqr[j] << endl;
			indm[i][j] = ret_indexes[j];
		}

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index_bg.index->findNeighbors(resultSet, &fv_unk[i][0], nanoflann::SearchParams(10));	
		for (int j = num_results; j < 2*num_results; j++){
			// cout << j-num_results<<" "<<i<<" ret_index["<<j<<"]=" << ret_indexes[j-num_results] << " out_dist_sqr=" << out_dists_sqr[j-num_results] << endl;
			indm[i][j] = ret_indexes[j-num_results];
		}
	}
	// cout<<indm.size()<<endl;
}



SparseMatrix<double> lle_KtoU(my_vector_of_vectors_t& indm, my_vector_of_vectors_t& fv_unk, 
	my_vector_of_vectors_t& fv_fg, my_vector_of_vectors_t& fv_bg, double eps, Mat& wf){
	
	int k = indm[0].size(); //number of neighbours that we are considering 
	int n = indm.size(); //number of unknown pixels

	int N = fv_unk.size()+fv_fg.size()+fv_bg.size();
	SparseMatrix<double> Wku(N,N), H(N,N);
	typedef Triplet<double> T;
    vector<T> triplets, tripletsH;
    triplets.reserve(k*n);
    tripletsH.reserve(n);

	Mat C(14, 14, DataType<double>::type), rhs(14, 1, DataType<double>::type), Z(3, 14, DataType<double>::type), weights(14, 1, DataType<double>::type);
	C = 0;
	rhs = 1;
	// cout<<k<<" "<<n<<endl;
	for(int i = 0; i < n; i++){
		// filling values in Z
		int j, p, index_nbr;
		for(j = 0; j < k/2; j++){
			index_nbr = indm[i][j];
			for(p = 0; p < dim-2; p++)
				Z.at<double>(p, j) = fv_fg[index_nbr][p] - fv_unk[i][p];
		}

		for(j = k/2; j < k; j++){
			index_nbr = indm[i][j];
			for(p = 0; p < dim-2; p++)
				Z.at<double>(p, j) = fv_bg[index_nbr][p] - fv_unk[i][p];
		}
		// cout<<"ours\n";
		// cout<<Z<<endl<<endl<<Z1<<endl;
		// exit(0);


		// adding some constant to ensure invertible matrices
		// C = Z.transpose()*Z;
		// C.diagonal().array() += eps;
		// weights = C.ldlt().solve(rhs);
		// weights /= weights.sum();
		// // cout<<weights<<endl;
		// wcm[i].resize(k);

		

		transpose(Z,C);	
		C = C*Z;
		for(int p = 0; p < k; p++)
			C.at<double>(p,p) += eps;
		// cout<<"determinant: "<<determinant(C)<<endl;
		solve(C, rhs, weights, DECOMP_CHOLESKY);
		double sum = 0;

		for(int j = 0; j < k; j++)
			sum += weights.at<double>(j,0);
		// cout<<"SUM:"<<sum;exit(0);
		int nbr_ind; 
		for(int j = 0; j < k/2; j++){
			weights.at<double>(j,0) /= sum;
			// nbr_ind = indm[i][j];
			// triplets.push_back(T(ind[{i,0}], ind[{nbr_ind, 1}], weights.at<double>(j,0)));
		}

		for(int j = k/2; j < k; j++){
			weights.at<double>(j,0) /= sum;
			// nbr_ind = indm[i][j];
			// triplets.push_back(T(ind[{i,0}], ind[{nbr_ind, 2}], weights.at<double>(j,0)));
		}


		// calculating confidence values
		double fweight = 0, bweight = 0, nu = 0; 
		double fcol[3], bcol[3]; 
		for(j = 0; j < 3; j++){
			fcol[j] = 0;
			bcol[j] = 0;
		}
		for(j = 0; j < k/2; j++){
			fweight += weights.at<double>(j,0);
			index_nbr = indm[i][j];
			for(p = 0; p < dim-2; p++)
				fcol[p] += weights.at<double>(j,0) * fv_fg[index_nbr][p];
		}
		
		for(j = k/2; j < k; j++){
			bweight += weights.at<double>(j,0);
			index_nbr = indm[i][j];
			for(p = 0; p < dim-2; p++)
				bcol[p] += weights.at<double>(j,0) * fv_bg[index_nbr][p];
		}
		
		double norm;
		for(j = 0; j < 3; j++){
			norm = fcol[j]/fweight - bcol[j]/bweight;
			nu += norm * norm;
		}

		// cout<<fweight<<" "<<bweight<<" "<<fweight+bweight<<" "<<nu<<endl;
		nu /= 3;
		tripletsH.push_back(T(ind[{i,0}], ind[{i,0}], nu));
		wf.at<uchar>(ind[{i,0}], 0) = fweight;
		
	}

	// Wku.setFromTriplets(triplets.begin(), triplets.end());
	H.setFromTriplets(tripletsH.begin(), tripletsH.end());
	return H;
}


SparseMatrix<double> KtoU(Mat& image, Mat& tmap, Mat& wf){
	my_vector_of_vectors_t fv_fg, fv_bg, fv_unk, indm, Euu;
	
    int i, j;
    kdtree_KtoU(image, tmap, indm, fv_unk, fv_fg, fv_bg);
	// cout<<"KD Tree done"<<endl;
	double eps = 0.001;
	// cout<<"lle done"<<endl;
	SparseMatrix<double> H = lle_KtoU(indm, fv_unk, fv_fg, fv_bg, eps, wf);
	cout<<"KToU Done"<<endl;
	return H;
}


/*
int main()
{
	Mat image,tmap;
	string img_path = "../../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);
 	SparseMatrix<double> ret = KtoU(image, tmap);

}
*/

// #endif