//Calonder descriptor sample
#include <stdio.h>

#if 0
#include <cxcore.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <vector>
using namespace std;

// Number of training points (set to -1 to use all points)
const int n_points = -1;

//Draw the border of projection of train image calculed by averaging detected correspondences
const bool draw_border = true;

void cvmSet6(CvMat* m, int row, int col, float val1, float val2, float val3, float val4, float val5, float val6)
{
    cvmSet(m, row, col, val1);
    cvmSet(m, row, col + 1, val2);
    cvmSet(m, row, col + 2, val3);
    cvmSet(m, row, col + 3, val4);
    cvmSet(m, row, col + 4, val5);
    cvmSet(m, row, col + 5, val6);
}

void FindAffineTransform(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* affine)
{
    int eq_num = 2*(int)p1.size();
    CvMat* A = cvCreateMat(eq_num, 6, CV_32FC1);
    CvMat* B = cvCreateMat(eq_num, 1, CV_32FC1);
    CvMat* X = cvCreateMat(6, 1, CV_32FC1);
    
    for(int i = 0; i < (int)p1.size(); i++)
    {
        cvmSet6(A, 2*i, 0, p1[i].x, p1[i].y, 1, 0, 0, 0);
        cvmSet6(A, 2*i + 1, 0, 0, 0, 0, p1[i].x, p1[i].y, 1);
        cvmSet(B, 2*i, 0, p2[i].x);
        cvmSet(B, 2*i + 1, 0, p2[i].y);
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

void MapVectorAffine(const vector<CvPoint>& p1, vector<CvPoint>& p2, CvMat* transform)
{
    float a = cvmGet(transform, 0, 0);
    float b = cvmGet(transform, 0, 1);
    float c = cvmGet(transform, 0, 2);
    float d = cvmGet(transform, 1, 0);
    float e = cvmGet(transform, 1, 1);
    float f = cvmGet(transform, 1, 2);
    
    for(int i = 0; i < (int)p1.size(); i++)
    {
        float x = a*p1[i].x + b*p1[i].y + c;
        float y = d*p1[i].x + e*p1[i].y + f;
        p2.push_back(cvPoint(x, y));
    }
}


float CalcAffineReprojectionError(const vector<CvPoint>& p1, const vector<CvPoint>& p2, CvMat* transform)
{
    vector<CvPoint> mapped_p1;
    MapVectorAffine(p1, mapped_p1, transform);
    float error = 0;
    for(int i = 0; i < (int)p2.size(); i++)
    {
        error += ((p2[i].x - mapped_p1[i].x)*(p2[i].x - mapped_p1[i].x)+(p2[i].y - mapped_p1[i].y)*(p2[i].y - mapped_p1[i].y));
    }
    
    error /= p2.size();
    
    return error;
}
#endif

int main( int, char** )
{
	printf("calonder_sample is under construction\n");
	return 0;

#if 0
	IplImage* test_image;
	IplImage* train_image;
	if (argc < 3)
	{
		
		test_image = cvLoadImage("box_in_scene.png",0);
		train_image = cvLoadImage("box.png ",0);
		if (!test_image || !train_image)
		{
			printf("Usage: calonder_sample <train_image> <test_image>");
			return 0;
		}
	}
	else
	{
		test_image = cvLoadImage(argv[2],0);
		train_image = cvLoadImage(argv[1],0);
	}





	if (!train_image)
	{
		printf("Unable to load train image\n");
		return 0;
	}

	if (!test_image)
	{
		printf("Unable to load test image\n");
		return 0;
	}



	CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq *objectKeypoints = 0, *objectDescriptors = 0;
    CvSeq *imageKeypoints = 0, *imageDescriptors = 0;
    CvSURFParams params = cvSURFParams(500, 1);
	cvExtractSURF( test_image, 0, &imageKeypoints, &imageDescriptors, storage, params );
	cvExtractSURF( train_image, 0, &objectKeypoints, &objectDescriptors, storage, params );


	cv::RTreeClassifier detector;
	int patch_width = cv::PATCH_SIZE;
	int patch_height = cv::PATCH_SIZE;
	vector<cv::BaseKeypoint> base_set;
	int i=0;
	CvSURFPoint* point;


	for (i=0;i<(n_points > 0 ? n_points : objectKeypoints->total);i++)
	{
		point=(CvSURFPoint*)cvGetSeqElem(objectKeypoints,i);
		base_set.push_back(cv::BaseKeypoint(point->pt.x,point->pt.y,train_image));
	}

	//Detector training
    cv::RNG rng( cvGetTickCount() );
	cv::PatchGenerator gen(0,255,2,false,0.7,1.3,-CV_PI/3,CV_PI/3,-CV_PI/3,CV_PI/3);

	printf("RTree Classifier training...\n");
	detector.train(base_set,rng,gen,24,cv::DEFAULT_DEPTH,2000,(int)base_set.size(),detector.DEFAULT_NUM_QUANT_BITS);
	printf("Done\n");

	float* signature = new float[detector.original_num_classes()];
	float* best_corr;
	int* best_corr_idx;
	if (imageKeypoints->total > 0)
	{
		best_corr = new float[imageKeypoints->total];
		best_corr_idx = new int[imageKeypoints->total];
	}

	for(i=0; i < imageKeypoints->total; i++)
	{
		point=(CvSURFPoint*)cvGetSeqElem(imageKeypoints,i);
		int part_idx = -1;
		float prob = 0.0f;


		CvRect roi = cvRect((int)(point->pt.x) - patch_width/2,(int)(point->pt.y) - patch_height/2, patch_width, patch_height);
		cvSetImageROI(test_image, roi);
		roi = cvGetImageROI(test_image);
		if(roi.width != patch_width || roi.height != patch_height)
		{
			best_corr_idx[i] = part_idx;
			best_corr[i] = prob;
		}
		else
		{
			cvSetImageROI(test_image, roi);
			IplImage* roi_image = cvCreateImage(cvSize(roi.width, roi.height), test_image->depth, test_image->nChannels);
			cvCopy(test_image,roi_image);

			detector.getSignature(roi_image, signature);


			for (int j = 0; j< detector.original_num_classes();j++)
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
	vector<CvPoint> object;
	vector<CvPoint> features;

	for (int j=0;j<objectKeypoints->total;j++)
	{
		float prob = 0.0f;
		int idx = -1;
		for (i = 0; i<imageKeypoints->total;i++)
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
			point=(CvSURFPoint*)cvGetSeqElem(objectKeypoints,j);
			object.push_back(cvPoint((int)point->pt.x,(int)point->pt.y));
			point=(CvSURFPoint*)cvGetSeqElem(imageKeypoints,idx);
			features.push_back(cvPoint((int)point->pt.x,(int)point->pt.y));
		}
	}
	if ((int)object.size() > 3)
	{
		CvMat* affine = cvCreateMat(2, 3, CV_32FC1);
		FindAffineTransform(object,features,affine);

		vector<CvPoint> corners;
		vector<CvPoint> mapped_corners;
		corners.push_back(cvPoint(0,0));
		corners.push_back(cvPoint(0,train_image->height));
		corners.push_back(cvPoint(train_image->width,0));
		corners.push_back(cvPoint(train_image->width,train_image->height));
		MapVectorAffine(corners,mapped_corners,affine);

		//Drawing the result
		IplImage* result = cvCreateImage(cvSize(test_image->width > train_image->width ? test_image->width : train_image->width,
			train_image->height + test_image->height),
			test_image->depth, test_image->nChannels);
		cvSetImageROI(result,cvRect(0,0,train_image->width, train_image->height));
		cvCopy(train_image,result);
		cvResetImageROI(result);
		cvSetImageROI(result,cvRect(0,train_image->height,test_image->width, test_image->height));
		cvCopy(test_image,result);
		cvResetImageROI(result);

		for (int i=0;i<(int)features.size();i++)
		{
			cvLine(result,object[i],cvPoint(features[i].x,features[i].y+train_image->height),cvScalar(255));
		}

		if (draw_border)
		{
			cvLine(result,cvPoint(mapped_corners[0].x, mapped_corners[0].y+train_image->height),
				cvPoint(mapped_corners[1].x, mapped_corners[1].y+train_image->height),cvScalar(150),3);
			cvLine(result,cvPoint(mapped_corners[0].x, mapped_corners[0].y+train_image->height),
				cvPoint(mapped_corners[2].x, mapped_corners[2].y+train_image->height),cvScalar(150),3);
			cvLine(result,cvPoint(mapped_corners[1].x, mapped_corners[1].y+train_image->height),
				cvPoint(mapped_corners[3].x, mapped_corners[3].y+train_image->height),cvScalar(150),3);
			cvLine(result,cvPoint(mapped_corners[2].x, mapped_corners[2].y+train_image->height),
				cvPoint(mapped_corners[3].x, mapped_corners[3].y+train_image->height),cvScalar(150),3);
		}

		cvSaveImage("Result.jpg",result);
		cvNamedWindow("Result",0);
		cvShowImage("Result",result);
		cvWaitKey();
		cvReleaseMat(&affine);
		cvReleaseImage(&result);
	}
	else
	{
		printf("Unable to find correspondence\n");
	}


	
	
	if (signature)
		delete[] signature;
	if (best_corr)
		delete[] best_corr;
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&train_image);
	cvReleaseImage(&test_image);

	return 0;
#endif
}
