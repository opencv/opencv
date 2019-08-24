// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
using namespace nanoflann; 
using namespace std;
using namespace cv;


typedef vector<vector<double>> my_vector_of_vectors_t;
typedef vector<set<int, greater<int>>> my_vector_of_set_t;
typedef vector<Mat> my_vector_of_Mat; 
typedef vector<pair<int, int>> my_vector_of_pair;

my_vector_of_vectors_t fv_unk, fv_fg, fv_bg; 
my_vector_of_Mat unkmean, fgmean, bgmean, unkcov, fgcov, bgcov;

// void type2str(int type) {
//   string r;

//   uchar depth = type & CV_MAT_DEPTH_MASK;
//   uchar chans = 1 + (type >> CV_CN_SHIFT);

//   switch ( depth ) {
//     case CV_8U:  r = "8U"; break;
//     case CV_8S:  r = "8S"; break;
//     case CV_16U: r = "16U"; break;
//     case CV_16S: r = "16S"; break;
//     case CV_32S: r = "32S"; break;
//     case CV_32F: r = "32F"; break;
//     case CV_64F: r = "64F"; break;
//     default:     r = "User"; break;
//   }

//   r += "C";
//   r += (chans+'0');
//   cout<<r<<endl;
//   // return r;
// }

// void show(Mat& image){
//     namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//     imshow( "Display window", image );                   // Show our image inside it.
//     waitKey(0);                                          // Wait for a keystroke in the window
// }

double l2norm(int x1, int y1, int x2, int y2){
	return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

void generateMean(Mat &img, Mat &tmap, my_vector_of_pair &map){
	// CV_Assert(img.depth() == CV_8U);

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;
	int win_size = 1;
	int num_win = (win_size*2 + 1)*(win_size*2 + 1); //number of pixels in window


	int fg = 0, bg = 0, unk = 0, c1 = 0, c2 = 0, c3 = 0;
	int i,j,k;
	for(i = win_size; i < img.rows-win_size; i++)
		for(j = win_size; j < img.cols-win_size; j++){
			float pix = tmap.at<uchar>(i,j);
    		if(pix == 128)
    			unk++;
    		else if(pix > 200)
    			fg++;
    		else bg++;
		}

	fv_fg.resize(fg); fgmean.resize(fg); fgcov.resize(fg); 
	fv_bg.resize(bg); bgmean.resize(bg); bgcov.resize(bg);
	fv_unk.resize(unk); unkmean.resize(unk); unkcov.resize(unk); map.resize(unk); 

	for(i = win_size; i < img.rows-win_size; i++){
		for(j = win_size; j < img.cols-win_size; j++){

			float pix = tmap.at<uchar>(i,j);

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

			if(pix == 128){
				fv_unk[c1].resize(3);		
				fv_unk[c1][0] = sum1/num_win;
				fv_unk[c1][1] = sum2/num_win;
				fv_unk[c1][2] = sum3/num_win;
				unkmean[c1] = win_mean; 
				unkcov[c1] = covariance;
				map[c1] = {i, j};
				if(c1 == 0){
					cout<<i<<" "<<j<<endl<<endl;
				}
				c1++;

			}
			else if(pix < 10){
				fv_bg[c2].resize(3);		
				fv_bg[c2][0] = sum1/num_win;
				fv_bg[c2][1] = sum2/num_win;
				fv_bg[c2][2] = sum3/num_win;
				bgmean[c2] = win_mean; 
				bgcov[c2] = covariance;
				c2++;
			}
			else{
				fv_fg[c3].resize(3);		
				fv_fg[c3][0] = sum1/num_win;
				fv_fg[c3][1] = sum2/num_win;
				fv_fg[c3][2] = sum3/num_win;
				fgmean[c3] = win_mean; 
				fgcov[c3] = covariance;
				c3++;
			}
			

			//Bhattacharya distance 
		}
	}
}

void findNearestNbr(my_vector_of_vectors_t& indm){

	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index_fg(3 /*dim*/, fv_fg, 10 /* max leaf */ );
	mat_index_fg.index->buildIndex();

	my_kd_tree_t mat_index_bg(3 /*dim*/, fv_bg, 10 /* max leaf */ );
	mat_index_bg.index->buildIndex();

	// do a knn search 20 nbrs
	const size_t num_results = 20; 

	int N = fv_unk.size();

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);

	indm.resize(N);
	int i = 0;
	for(i = 0; i < fv_unk.size(); i++){
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
}


double Bhattacharya(Mat mean1, Mat mean2, Mat cov1, Mat cov2){
	Mat sigma = (cov1 + cov2)/2;
	double denom = sqrt(determinant(cov1)*determinant(cov2));
	Mat x = (0.125)*(mean1-mean2)*sigma.inv()*(mean1-mean2).t();
	return x.at<double>(0,0) + (0.5)*log(determinant(sigma)/denom);
}

void trimming(Mat &img, Mat &tmap, Mat &new_tmap, bool post){
  
    int nRows = img.rows; 
    int nCols = img.cols; 

    //Shahrian implementation 
    // Mat new_tmap = tmap.clone(); 
    int thresh = 9;
    int win_size = thresh-1;

 	int unk_count = 0; 
	int i,j,k; 
	double cnorm, dist;
	for( i = 0; i < nRows; ++i){
		for ( j = 0; j < nCols; ++j){
			float pix = tmap.at<uchar>(i,j);
    		if(float(tmap.at<uchar>(i,j)) == 128){					
				bool assigned = false;
				for(int p = max(i-win_size, 0); p < min(i+win_size+1, nRows) && !assigned; p++){
					for(int q = max(j-win_size, 0); q < min(j+win_size+1, nCols) && !assigned; q++){
						float pix_nbr = tmap.at<uchar>(p,q);
						if(pix_nbr != 128){
							dist = l2norm(p, q, i, j);
							if(dist < thresh){
								cv::Vec3b diff = img.at<cv::Vec3b>(p,q) - img.at<cv::Vec3b>(i,j); 
								cnorm = diff[0] * diff[0];
								cnorm += diff[1] * diff[1];
								cnorm += diff[2] * diff[2]; 
								cnorm = sqrt(cnorm);
								if(cnorm < (thresh-dist)){
									// set as foreground/background
									new_tmap.at<uchar>(i,j) = pix_nbr;
									assigned = true;
								}	
							}
							else continue;	
						}	
					}
				}
    		}
		}
	}
	imwrite("2.png", new_tmap);

	if(post)
		return;

	//patch trimming
	Mat mean1, mean2, cov1, cov2; 
	int channels = img.channels();
	my_vector_of_vectors_t indm;
	my_vector_of_pair map; 

	generateMean(img, new_tmap, map);
	findNearestNbr(indm);

	k = indm[0].size(); //number of neighbours that we are considering 
	int n = indm.size(); //number of unknown pixels
	
	double minfg = 100, minbg = 100; //set to random values
	float tauc = 0.25, tauf = 0.9;
	int imgi, imgj;
	for(int i = 0; i < n; i++){
		// filling values in Z
		int j, p, index_nbr;
		mean1 = unkmean[i];
		cov1 = unkcov[i];

		for(j = 0; j < k/2; j++){
			index_nbr = indm[i][j];
			mean2 = fgmean[index_nbr];
			cov2 = fgcov[index_nbr]; 
			minfg = min(minfg, Bhattacharya(mean1, mean2, cov1, cov2));
		}

		for(j = k/2; j < k; j++){
			index_nbr = indm[i][j];
			mean2 = bgmean[index_nbr];
			cov2 = bgcov[index_nbr];
			minbg = min(minbg, Bhattacharya(mean1, mean2, cov1, cov2));
		}
		// cout<<"done****"<<endl;
		
		
		imgi = map[i].first; imgj = map[i].second;
		// cout<<minfg<<" "<<minbg<<endl;
		if(minfg < tauc && minbg > tauf){
			new_tmap.at<uchar>(imgi,imgj) = 255;	//fg
			cout<<"fg"<<endl;
		}
		else if(minbg < tauc && minfg > tauf){
			new_tmap.at<uchar>(imgi,imgj) = 0;	//bg
			cout<<"bg"<<endl;	
		}
		// else remain unknown 
	}

	imwrite("1.png", tmap);
	imwrite("3.png", new_tmap);
}



/*

int main(){

	Mat img,tmap;
	// my_vector_of_vectors_t samples, indm, Euu;
	string img_path = "../../data/input_lowres/net.png";
	img = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file

    string tmap_path = "../../data/trimap_lowres/Trimap1/net.png";
    tmap = imread(tmap_path, CV_LOAD_IMAGE_GRAYSCALE);

    Mat new_tmap = tmap.clone();
    trimming(tmap, tmap, new_tmap, tmap, true); 
}

*/