// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifndef local_info
// #define local_info

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "Eigen/Sparse"
using namespace Eigen;
using namespace std;
using namespace cv;

// const int dim = 5;

// void show(Mat& image){
//     namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//     imshow( "Display window", image );                   // Show our image inside it.
//     waitKey(0);                                          // Wait for a keystroke in the window
// }


void local_info(Mat& img, Mat& tmap, SparseMatrix<double>& Wl, SparseMatrix<double>& Dl){


	float eps = 0.001;
	int win_size = 1; 

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;
	Mat unk_img;
	unk_img.create(nRows, nCols, CV_32FC1);
	// Mat unk_img1 = Mat::zeros(cv::Size(2,5), CV_8U);	
	// cout<<unk_img.size<<endl;exit(0);

	// cout<<nRows<<" "<<nCols<<endl;
	int c1 = 0;	
	for(int i = 0; i < nRows; ++i)
		for (int j = 0; j < nCols; ++j){
			float pix = tmap.at<uchar>(i,j);
    		if(pix == 128){							//collection of unknown pixels samples
				unk_img.at<float>(i,j) = 255;
			}
		}

	// cout<<unk_img<<endl;
	Mat element = getStructuringElement(MORPH_RECT, Size(2*win_size + 1, 2*win_size+1));
  	/// Apply the dilation operation
  	Mat dilation_dst; 
	dilate(unk_img, dilation_dst, element);
	// show(tmap);
	// show(dilation_dst);


	int num_win = (win_size*2 + 1)*(win_size*2 + 1); //number of pixels in window
	int num_win_sq = num_win*num_win;
	int N = img.rows*img.cols;

	// Leaving this computation ---can edit 
	// int sparse_ent = 0; //count of sparse entries
	//  = (nRows - 2 * win_size) * (nCols - 2 * win_size);
 	// tlen += sum(dilation_dst);
	// tlen *= neb_size_square;

	// SparseMatrix<double> Wl(N,N), Dl(N,N);
	typedef Triplet<double> T;
    vector<T> triplets, td;

    int x[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int y[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

	int i, j; 
	for(i = win_size; i < img.rows-win_size; i++){
		for(j = win_size; j < img.cols-win_size; j++){

			if ((int)dilation_dst.at<uchar>(i, j) == 0) {
        		continue;
      		}

			// extract the window out of image
			Mat win = img.rowRange(i-win_size, i+win_size+1);
			win = win.colRange(j-win_size, j+win_size+1);
			Mat win_ravel = Mat::zeros(9, 3, CV_64F); //doubt ??
			double sum1 = 0;
			double sum2 = 0;
			double sum3 = 0; 

			int c = 0;
			for(int p = 0; p < win_size*2+1; p++){
				for(int q = 0; q < win_size*2+1; q++){
					win_ravel.at<double>(c,0) = win.at<cv::Vec3b>(p,q)[0]/255.0;
					win_ravel.at<double>(c,1) = win.at<cv::Vec3b>(p,q)[1]/255.0;
					win_ravel.at<double>(c,2) = win.at<cv::Vec3b>(p,q)[2]/255.0;
					// cout<<double(win.at<cv::Vec3b>(p,q)[0])<<endl;
					// exit(0);
					sum1 += win.at<cv::Vec3b>(p,q)[0]/255.0;
					sum2 += win.at<cv::Vec3b>(p,q)[1]/255.0;
					sum3 += win.at<cv::Vec3b>(p,q)[2]/255.0;
					c++;
				}
			}
			win = win_ravel;

			Mat win_mean = Mat::zeros(1, 3, CV_64F);
			win_mean.at<double>(0,0) = sum1/num_win; 
			win_mean.at<double>(0,1) = sum2/num_win; 
			win_mean.at<double>(0,2) = sum3/num_win; 

			// calculate the covariance matrix 
      		Mat covariance = (win.t() * win / num_win) - (win_mean.t() * win_mean);

      		Mat I = Mat::eye(img.channels(), img.channels(), CV_64F);
      		Mat inv = (covariance + eps / num_win * I).inv();

      		Mat X = win - repeat(win_mean, num_win, 1);
      		Mat vals = (1 + X * inv * X.t()) / num_win;
      		vals = vals.reshape(0, num_win_sq);

      		int nbr_r, nbr_c; //nrb row and col 
      		for(int p = 0; p < num_win; p++){
      			for(int q = 0; q < num_win; q++){
      				nbr_r = i+x[p];
      				nbr_c = j+y[p];
      				triplets.push_back(T(nbr_r, nbr_c, vals.at<double>(p*num_win+q, 0)));
      				td.push_back(T(nbr_r, nbr_r, vals.at<double>(p*num_win+q, 0)));
      			}
      		}
      		// cout<<vals<<endl;

		}
	}

	Wl.setFromTriplets(triplets.begin(), triplets.end());
	Dl.setFromTriplets(triplets.begin(), triplets.end());
	cout<<"local_info DONE"<<endl;
	// return Wl;
}


/*
int main(){

	Mat image,tmap;
	// my_vector_of_vectors_t samples, indm, Euu;
	string img_path = "../../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    string tmap_path = "../../data/trimap_lowres/Trimap1/plasticbag.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);

    int N = image.rows*image.cols; 
    SparseMatrix<double> Wl(N,N), Dl(N,N);
    local_info(image, tmap, Wl, Dl);
 
}
*/

// #endif