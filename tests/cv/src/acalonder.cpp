/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "cvtest.h"

#if 0

#include "highgui.h"
#include <vector>
#include <string>
using namespace std;

using namespace cv;

class CV_CalonderTest : public CvTest
{
public:
    CV_CalonderTest();
    ~CV_CalonderTest();    
protected:    
    void run(int);


	void cvmSet6(CvMat* m, int row, int col, float val1, float val2, float val3, float val4, float val5, float val6);
	void FindAffineTransform(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* affine);
	void MapVectorAffine(const vector<CvPoint>& p1, vector<CvPoint>& p2, CvMat* transform);
	float CalcAffineReprojectionError(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* transform);
	void ExtractFeatures(const IplImage* image, vector<CvPoint>& points);
	void TrainDetector(RTreeClassifier& detector, int/* patch_size*/, const vector<CvPoint>& train_points,const IplImage* train_image, int n_keypoints = 0);
	void GetCorrespondences(const RTreeClassifier& detector, int patch_size,
						const vector<CvPoint>& objectKeypoints, const vector<CvPoint>& imageKeypoints, const IplImage* image,
						vector<CvPoint>& object, vector<CvPoint>& features);

	// Scales the source image (x and y) and rotate to the angle (Positive values mean counter-clockwise rotation) 
	void RotateAndScale(const IplImage* src, IplImage* dst, float angle, float scale_x, float scale_y);
	// Scales the source image point and rotate to the angle (Positive values mean counter-clockwise rotation) 
	void RotateAndScale(const CvPoint& src, CvPoint& dst, const CvPoint& center, float angle, float scale_x, float scale_y);
	float RunTestsSeries(const IplImage* train_image, vector<CvPoint>& keypoints/*, bool drawResults = false*/);
	//returns 1 in the case of success, 0 otherwise
	int SaveKeypoints(const vector<CvPoint>& points, const char* path);
	////returns 1 in the case of success, 0 otherwise
	int LoadKeypoints(vector<CvPoint>& points, const char* path);

	void ExtractKeypointSignatures(const IplImage* test_image, int patch_size, const RTreeClassifier& detector, const vector<CvPoint>& keypoints, vector<vector<float> >& signatures);
	//returns 1 in the case of success, 0 otherwise
	int SaveKeypointSignatures(const char* path, const vector<vector<float> >& signatures);
	//returns 1 in the case of success, 0 otherwise
	int LoadKeypointSignatures(const char* path, vector<vector<float> >& signatures);

	//returns 1 in the case signatures are identical, 0 otherwise
	int CompareSignatures(const vector<vector<float> > & signatures1, const vector<vector<float> >& signatures2);


};

void CV_CalonderTest::cvmSet6(CvMat* m, int row, int col, float val1, float val2, float val3, float val4, float val5, float val6)
{
    cvmSet(m, row, col, val1);
    cvmSet(m, row, col + 1, val2);
    cvmSet(m, row, col + 2, val3);
    cvmSet(m, row, col + 3, val4);
    cvmSet(m, row, col + 4, val5);
    cvmSet(m, row, col + 5, val6);
}

void CV_CalonderTest::FindAffineTransform(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* affine)
{
    int eq_num = 2*(int)p1.size();
    CvMat* A = cvCreateMat(eq_num, 6, CV_32FC1);
    CvMat* B = cvCreateMat(eq_num, 1, CV_32FC1);
    CvMat* X = cvCreateMat(6, 1, CV_32FC1);
    
    for(int i = 0; i < (int)p1.size(); i++)
    {
        cvmSet6(A, 2*i, 0, (float)p1[i].x, (float)p1[i].y, 1, 0, 0, 0);
        cvmSet6(A, 2*i + 1, 0, 0, 0, 0, (float)p1[i].x, (float)p1[i].y, 1);
        cvmSet(B, 2*i, 0, (double)p2[i].x);
        cvmSet(B, 2*i + 1, 0, (double)p2[i].y);
    }
    
    cvSolve(A, B, X, CV_SVD);
    
    cvmSet(affine, 0, 0, cvmGet(X, 0, 0));
    cvmSet(affine, 0, 1, cvmGet(X, 1, 0));
    cvmSet(affine, 0, 2, cvmGet(X, 2, 0));
    cvmSet(affine, 1, 0, cvmGet(X, 3, 0));
    cvmSet(affine, 1, 1, cvmGet(X, 4, 0));
    cvmSet(affine, 1, 2, cvmGet(X, 5, 0));
    
    cvReleaseMat(&A);
    cvReleaseMat(&B);
    cvReleaseMat(&X);
}

void CV_CalonderTest::MapVectorAffine(const vector<CvPoint>& p1, vector<CvPoint>& p2, CvMat* transform)
{
    double a = cvmGet(transform, 0, 0);
    double b = cvmGet(transform, 0, 1);
    double c = cvmGet(transform, 0, 2);
    double d = cvmGet(transform, 1, 0);
    double e = cvmGet(transform, 1, 1);
    double f = cvmGet(transform, 1, 2);
    
    for(int i = 0; i < (int)p1.size(); i++)
    {
        double x = a*p1[i].x + b*p1[i].y + c;
        double y = d*p1[i].x + e*p1[i].y + f;
        p2.push_back(cvPoint((int)x, (int)y));
    }
}


float CV_CalonderTest::CalcAffineReprojectionError(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* transform)
{
    vector<CvPoint> mapped_p1;
    MapVectorAffine(p1, mapped_p1, transform);
    float error = 0;
    for(int i = 0; i < (int)p2.size(); i++)
    {
        //float l = length(p2[i] - mapped_p1[i]);
        error += ((p2[i].x - mapped_p1[i].x)*(p2[i].x - mapped_p1[i].x)+(p2[i].y - mapped_p1[i].y)*(p2[i].y - mapped_p1[i].y));
    }
    
    error /= p2.size();
    
    return error;
}

void CV_CalonderTest::ExtractFeatures(const IplImage* image, vector<CvPoint>& points)
{
	points.clear();
	CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *keypoints = 0, *descriptors = 0;
    CvSURFParams params = cvSURFParams(1000, 1);
	cvExtractSURF( image, 0, &keypoints, &descriptors, storage, params );

	
	CvSURFPoint* point;
	for (int i=0;i<keypoints->total;i++)
	{
		point=(CvSURFPoint*)cvGetSeqElem(keypoints,i);
		points.push_back(cvPoint((int)(point->pt.x),(int)(point->pt.y)));
	}
	cvReleaseMemStorage(&storage);
}

void CV_CalonderTest::TrainDetector(RTreeClassifier& detector, int/* patch_size*/, const vector<CvPoint>& train_points,const IplImage* train_image, int n_keypoints)
{
	vector<BaseKeypoint> base_set;
	int n = (int)(train_points.size());
	if (n_keypoints)
		n = n_keypoints;
	for (int i=0;i<n;i++)
	{
		base_set.push_back(BaseKeypoint(train_points[i].x,train_points[i].y,const_cast<IplImage*>(train_image)));
	}

	//Detector training
	//CvRNG r = cvRNG(1);
	RNG rng( cvRandInt(this->ts->get_rng()));
    PatchGenerator gen(0,255,2,false,0.7,1.3,-CV_PI/3,CV_PI/3,-CV_PI/3,CV_PI/3);

	//int64 t0 = cvGetTickCount();
	detector.train(base_set,rng,gen,6,DEFAULT_DEPTH,3000,(int)base_set.size(),detector.DEFAULT_NUM_QUANT_BITS,false);
	//int64 t1 = cvGetTickCount();
	//printf("Train: %f s\n",(float)(t1-t0)/cvGetTickFrequency()*1e-6);


}

void CV_CalonderTest::GetCorrespondences(const RTreeClassifier& detector, int patch_size,
						const vector<CvPoint>& objectKeypoints, const vector<CvPoint>& imageKeypoints, const IplImage* image,
						vector<CvPoint>& object, vector<CvPoint>& features)
{
	IplImage* test_image = cvCloneImage(image);
	object.clear();
	features.clear();

	float* signature = new float[(const_cast<RTreeClassifier&>(detector)).original_num_classes()];
	float* best_corr;
	int* best_corr_idx;
	if (imageKeypoints.size() > 0)
	{
		best_corr = new float[(int)imageKeypoints.size()];
		best_corr_idx = new int[(int)imageKeypoints.size()];


	for(int i=0; i < (int)imageKeypoints.size(); i++)
	{
		int part_idx = -1;
		float prob = 0.0f;
		//CvPoint center = cvPoint((int)(imageKeypoints[i].x),(int)(imageKeypoints[i].y));

		CvRect roi = cvRect((int)(imageKeypoints[i].x) - patch_size/2,(int)(imageKeypoints[i].y) - patch_size/2, patch_size, patch_size);
		cvSetImageROI(test_image, roi);
		roi = cvGetImageROI(test_image);
		if(roi.width != patch_size || roi.height != patch_size)
		{
			best_corr_idx[i] = part_idx;
			best_corr[i] = prob;
		}
		else
		{
			cvSetImageROI(test_image, roi);
			IplImage* roi_image = cvCreateImage(cvSize(roi.width, roi.height), test_image->depth, test_image->nChannels);
			cvCopy(test_image,roi_image);

			(const_cast<RTreeClassifier&>(detector)).getSignature(roi_image, signature);


			for (int j = 0; j< (const_cast<RTreeClassifier&>(detector)).original_num_classes();j++)
			{
				if (prob < signature[j])
				{
					part_idx = j;
					prob = signature[j];
				}
			}

			best_corr_idx[i] = part_idx;
			best_corr[i] = prob;

			
			if (roi_image)
				cvReleaseImage(&roi_image);
		}
		cvResetImageROI(test_image);
	}

	float min_prob = 0.0f;

	for (int j=0;j<(int)objectKeypoints.size();j++)
	{
		float prob = 0.0f;
		int idx = -1;
		for (int i = 0; i<(int)imageKeypoints.size();i++)
		{
			if ((best_corr_idx[i]!=j)||(best_corr[i] < min_prob))
				continue;

			if (best_corr[i] > prob)
			{
				prob = best_corr[i];
				idx = i;
			}
		}
		if (idx >=0)
		{
			object.push_back(objectKeypoints[j]);
			features.push_back(imageKeypoints[idx]);
		}
	}

	if (best_corr)
		delete[] best_corr;
	if (best_corr_idx)
		delete[] best_corr_idx;
		}
	cvReleaseImage(&test_image);
	if (signature)
		delete[] signature;
}


// Scales the source image (x and y) and rotate to the angle (Positive values mean counter-clockwise rotation) 
void CV_CalonderTest::RotateAndScale(const IplImage* src, IplImage* dst, float angle, float scale_x, float scale_y)
{
	IplImage* temp = cvCreateImage(cvSize((int)(src->width*scale_x),(int)(src->height*scale_y)),src->depth,src->nChannels);

	cvResize(src,temp);

	CvMat* transform = cvCreateMat(2,3,CV_32FC1);
	cv2DRotationMatrix(cvPoint2D32f(((double)temp->width)/2,((double)temp->height)/2), angle*180/CV_PI,
		1.0f, transform );

	cvWarpAffine( temp, dst, transform,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
	cvReleaseImage(&temp);
	cvReleaseMat(&transform);
}

// Scales the source image point and rotate to the angle (Positive values mean counter-clockwise rotation) 
void CV_CalonderTest::RotateAndScale(const CvPoint& src, CvPoint& dst, const CvPoint& center, float angle, float scale_x, float scale_y)
{
	CvPoint temp;
	temp.x = (int)(src.x*scale_x);
	temp.y = (int)(src.y*scale_y);

	CvMat* transform = cvCreateMat(2,3,CV_32FC1);
	cv2DRotationMatrix(cvPoint2D32f((double)center.x*scale_x,(double)center.y*scale_y), angle*180/CV_PI,
		1.0f, transform );

    double a = cvmGet(transform, 0, 0);
    double b = cvmGet(transform, 0, 1);
    double c = cvmGet(transform, 0, 2);
    double d = cvmGet(transform, 1, 0);
    double e = cvmGet(transform, 1, 1);
    double f = cvmGet(transform, 1, 2);
    
    double x = a*temp.x + b*temp.y + c;
    double y = d*temp.x + e*temp.y + f;
    dst= cvPoint((int)x, (int)y);

	cvReleaseMat(&transform);
}

float CV_CalonderTest::RunTestsSeries(const IplImage* train_image, vector<CvPoint>& keypoints)
{
	float angles[] = {(float)-CV_PI/4,(float)CV_PI/4};
	float scales_x[] = {0.85f,1.15f};
	float scales_y[] = {0.85f,1.15f};
	int n_angles = 4;
	int n_scales_x = 3;
	int n_scales_y = 3;
	int accuracy = 4;
	int are_keypoints_loaded = (int)keypoints.size();

	int total_cases = n_angles*n_scales_x*n_scales_y;
	int n_case = 0;

	int length = max(train_image->width,train_image->height);
	int move_x = (int)(1.5*scales_x[0]*length);
	int move_y = (int)(1.5*scales_y[0]*length);
	IplImage* test_image = cvCreateImage(cvSize((int)(scales_x[1]*(move_x+length*1.5)),(int)(scales_y[1]*(move_y+length*1.5))),
				train_image->depth, train_image->nChannels);
	cvSet(test_image,cvScalar(0));

	cvSetImageROI(test_image,cvRect(move_x,move_y,train_image->width,train_image->height));
	cvCopy(train_image,test_image);
	cvResetImageROI(test_image);

	vector<CvPoint> objectKeypoints;
	if (!are_keypoints_loaded)
	{
		ExtractFeatures(train_image,objectKeypoints);
		for (int i=0;i<(int)objectKeypoints.size();i++)
		{
			keypoints.push_back(objectKeypoints[i]);
		}
	}
	else
	{
		for (int i=0;i<(int)keypoints.size();i++)
		{
			objectKeypoints.push_back(keypoints[i]);
		}
	}

	//Checking signatures are identical
	vector <vector<float> > signatures1;
	string signatures_path = string(ts->get_data_path()) + "calonder/signatures.txt";
	int can_load_signatures = LoadKeypointSignatures(signatures_path.c_str(),signatures1);
	// end of region

	RTreeClassifier detector;
	int patch_size = PATCH_SIZE;
	//this->update_progress(1,0,total_cases,5);
	TrainDetector(detector,patch_size,objectKeypoints,train_image,20);

	//Checking signatures are identical
	vector <vector<float> > signatures2;
	ExtractKeypointSignatures(train_image,patch_size,detector,objectKeypoints,signatures2);
	if (!can_load_signatures)
	{
		//SaveKeypointSignatures(signatures_path.c_str(),signatures2);
	}
	else
	{
	//	if (!CompareSignatures(signatures1,signatures2))
	//		return 0;
	}
	// end of region



	int points_total = 0;
	int points_correct = 0;
	

	vector<CvPoint> imageKeypoints;
	vector<CvPoint> object;
	vector<CvPoint> features;
	IplImage* temp = cvCloneImage(test_image);

	int progress = 0;


	//int64 t0 = cvGetTickCount();
	//printf("\n\n-----------\nTest started\n-----------\n");
	for (float angle = angles[0]; angle<=angles[1];angle+=(n_angles > 1 ?(angles[1]-angles[0])/n_angles : 1))
	{
		for (float scale_x = scales_x[0]; scale_x<=scales_x[1];scale_x+=(n_scales_x > 1 ? (scales_x[1]-scales_x[0])/n_scales_x : 1))
		{
			for (float scale_y = scales_y[0]; scale_y<=scales_y[1];scale_y+=(n_scales_y > 1 ? (scales_y[1]-scales_y[0])/n_scales_y : 1))
			{
				//printf("---\nAngle: %f, scaleX: %f, scaleY: %f\n", angle,scale_x,scale_y);
				cvSet(temp,cvScalar(0));
				imageKeypoints.clear();
				object.clear();
				features.clear();

				RotateAndScale(test_image,temp,angle,scale_x,scale_y);
				ExtractFeatures(temp,imageKeypoints);
				GetCorrespondences(detector,patch_size,objectKeypoints,imageKeypoints,temp,object,features);

				int correct = 0;
				CvPoint res;
				for (int i = 0; i< (int)object.size(); i++)
				{

					CvPoint current = object[i];
					current.x+=move_x;
					current.y+=move_y;
					RotateAndScale(current,res,cvPoint(temp->width/2,temp->height/2),angle,scale_x,scale_y);
					int dist = (res.x - features[i].x)*(res.x - features[i].x)+(res.y - features[i].y)*(res.y - features[i].y);
					if (dist < accuracy*accuracy)
						correct++;
				}
				//printf("Image points: %d\nCorrespondences found: %d/%d\n", (int)imageKeypoints.size(), correct, (int)object.size());
				points_correct+=correct;
				points_total+=(int)object.size();
				progress = update_progress( progress, n_case++, total_cases, 0 );
				//if (drawResults)
				//{					
				//	DrawResult(train_image, temp,object,features);
				//}				
			}
		}
	}
//	int64 t1 = cvGetTickCount();
	//printf("%f s\n",(float)(t1-t0)/cvGetTickFrequency()*1e-6);
	cvReleaseImage(&temp);
	cvReleaseImage(&test_image);
	//printf("\n\n-----------\nTest completed\n-----------\n");
	//printf("Total correspondences found: %d/%d\n", points_correct, points_total);
	//FILE* f = fopen("test_result.txt","w");
	//fprintf(f,"Total correspondences found: %d/%d\n", points_correct, points_total);
	//fclose(f);
	if (points_total < 1)
	{
		points_correct = 0;
		points_total = 1;
	}
	return (float)points_correct/(float)points_total;

}

CV_CalonderTest::CV_CalonderTest() : CvTest("calonder","RTreeClassifier")
{
}

CV_CalonderTest::~CV_CalonderTest() {}

int CV_CalonderTest::SaveKeypoints(const vector<CvPoint>& points, const char* path)
{
	FILE* f = fopen(path,"w");
	if (f==NULL)
	{
		return 0;
	}
	for (int i=0;i<(int)points.size();i++)
	{
		fprintf(f,"%d,%d\n",points[i].x,points[i].y);
	}
	fclose(f);
	return 1;
}

int CV_CalonderTest::LoadKeypoints(vector<CvPoint>& points, const char* path)
{
	FILE* f = fopen(path,"r");
	points.clear();

	if (f==NULL)
	{
		return 0;
	}
	
	while (!feof(f))
	{
		int x,y;
		fscanf(f,"%d,%d\n",&x,&y);
		points.push_back(cvPoint(x,y));
	}
	fclose(f);
	return 1;
}

void CV_CalonderTest::ExtractKeypointSignatures(const IplImage* test_image, int patch_size, const RTreeClassifier& detector, const vector<CvPoint>& keypoints, vector<vector<float> >& signatures)
{
	IplImage* _test_image = cvCloneImage(test_image);
	signatures.clear();

	float* signature = new float[(const_cast<RTreeClassifier&>(detector)).original_num_classes()];

	for (int i=0;i<(int)keypoints.size();i++)
	{
		CvRect roi = cvRect((int)(keypoints[i].x) - patch_size/2,(int)(keypoints[i].y) - patch_size/2, patch_size, patch_size);
		cvSetImageROI(_test_image, roi);
		roi = cvGetImageROI(_test_image);
		if(roi.width != patch_size || roi.height != patch_size)
		{
			continue;
		}

		cvSetImageROI(_test_image, roi);
		IplImage* roi_image = cvCreateImage(cvSize(roi.width, roi.height), _test_image->depth, _test_image->nChannels);
		cvCopy(_test_image,roi_image);

		(const_cast<RTreeClassifier&>(detector)).getSignature(roi_image, signature);
		
		vector<float> vec;

		for (int j=0;j<(const_cast<RTreeClassifier&>(detector)).original_num_classes();j++)
		{
			vec.push_back(signature[j]);
		}
		signatures.push_back(vec);

		cvReleaseImage(&roi_image);

	}

	delete[] signature;
	cvReleaseImage(&_test_image);
}




int CV_CalonderTest::SaveKeypointSignatures(const char* path, const vector<vector<float> >& signatures)
{
	FILE* f = fopen(path,"w");
	if (!f)
		return 0;

	for (int i=0;i<(int)signatures.size();i++)
	{
		for (int j=0;j<(int)signatures[i].size();j++)
		{
			fprintf(f,"%f",signatures[i][j]);
			if (j<((int)signatures[i].size()-1))
				fprintf(f,",");
		}
		if (i<((int)signatures.size()-1))
			fprintf(f,"\n");
	}
	fclose(f);

	return 1;
}

int CV_CalonderTest::LoadKeypointSignatures(const char* path, vector<vector<float> >& signatures)
{
	signatures.clear();
	FILE* f = fopen(path,"r");
	if (!f)
		return 0;

	char line[4096];
	vector<float> vec;
	char* tok;

	while(fgets(line,4096,f))
	{
		vec.clear();	
		float val;
		tok = strtok(line,",");
		if (tok)
		{
			sscanf(tok,"%f",&val);
			vec.push_back(val);
			tok = strtok(NULL,",");
			while (tok)
			{
				sscanf(tok,"%f",&val);
				vec.push_back(val);
				tok = strtok(NULL,",");
			}
			signatures.push_back(vec);
		}
	}

	fclose(f);
	return(1);
}

int CV_CalonderTest::CompareSignatures(const vector<vector<float> >& signatures1, const vector<vector<float> >& signatures2)
{
	if (signatures1.size() != signatures2.size())
	{
		return 0;
	}

	float accuracy = 0.05f;
	for (int i=0;i<(int)signatures1.size();i++)
	{
		if (signatures1[i].size() != signatures2[i].size())
		{
			return 0;
		}
		for (int j=0;j<(int)signatures1[i].size();j++)
		{
			if (abs(signatures1[i][j]-signatures2[i][j]) > accuracy)
				return 0;
		}
	}
	return 1;
}


void CV_CalonderTest::run( int /* start_from */)
{
	string train_image_path = string(ts->get_data_path()) + "calonder/baboon200.jpg";
	string train_keypoints_path = string(ts->get_data_path()) + "calonder/train_features.txt";
	IplImage* train_image = cvLoadImage(train_image_path.c_str(),0);

	if (!train_image)
	{
		ts->printf( CvTS::LOG, "Unable to open train image calonder/baboon200.jpg");
		ts->set_failed_test_info(CvTS::FAIL_MISSING_TEST_DATA);
		return;
	}



	// Testing rtree classifier
	float min_accuracy = 0.35f;
	vector<CvPoint> train_keypoints;
	train_keypoints.clear();
	float correctness;
	if (!LoadKeypoints(train_keypoints,train_keypoints_path.c_str()))
	{
		correctness = RunTestsSeries(train_image,train_keypoints);
		SaveKeypoints(train_keypoints,train_keypoints_path.c_str());
	}
	else
	{
		correctness = RunTestsSeries(train_image,train_keypoints);
	}
	if (correctness > min_accuracy)
		ts->set_failed_test_info(CvTS::OK);
	else
	{
		ts->set_failed_test_info(CvTS::FAIL_BAD_ACCURACY);
		ts->printf( CvTS::LOG, "Correct correspondences: %f, less than %f",correctness,min_accuracy);
	}
}

CV_CalonderTest calonder_test;

#endif
