/*
 *  one_way_sample.cpp
 *  outlet_detection
 *
 *  Created by Victor  Eruhimov on 8/5/09.
 *  Copyright 2009 Argus Corp. All rights reserved.
 *
 */

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

#include <string>

using namespace cv;

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, 
                              IplImage* img2, const vector<KeyPoint>& features2, const vector<int>& desc_idx);
void generatePCADescriptors(const char* img_path, const char* pca_low_filename, const char* pca_high_filename, 
                            const char* pca_desc_filename, CvSize patch_size);

int main(int argc, char** argv)
{    
    const char pca_high_filename[] = "pca_hr.yml";
    const char pca_low_filename[] = "pca_lr.yml";
    const char pca_desc_filename[] = "pca_descriptors.yml";
    const CvSize patch_size = cvSize(24, 24);
    const int pose_count = 50;
    
    if(argc != 3 && argc != 4)
    {
        printf("Format: \n./one_way_sample [path_to_samples] [image1] [image2]\n");
        printf("For example: ./one_way_sample ../../../opencv/samples/c scene_l.bmp scene_r.bmp\n");
        return 0;
    }
    
    std::string path_name = argv[1];
    std::string img1_name = path_name + "/" + std::string(argv[2]);
    std::string img2_name = path_name + "/" + std::string(argv[3]);

    CvFileStorage* fs = cvOpenFileStorage("pca_hr.yml", NULL, CV_STORAGE_READ);
    if(fs == NULL)
    {
        printf("PCA data is not found, starting training...\n");
        generatePCADescriptors(path_name.c_str(), pca_low_filename, pca_high_filename, pca_desc_filename, patch_size);
    }
    else
    {
        cvReleaseFileStorage(&fs);
    }
    
    
    printf("Reading the images...\n");
    IplImage* img1 = cvLoadImage(img1_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* img2 = cvLoadImage(img2_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    // extract keypoints from the first image
    vector<KeyPoint> keypoints1;
    SURF surf_extractor(5.0e3);
//    printf("Extracting keypoints\n");
    surf_extractor(img1, Mat(), keypoints1);
    printf("Extracted %d keypoints...\n", (int)keypoints1.size());

    printf("Training one way descriptors...");
    // create descriptors 
    OneWayDescriptorBase descriptors(patch_size, pose_count, ".", pca_low_filename, pca_high_filename, pca_desc_filename);
    descriptors.CreateDescriptorsFromImage(img1, keypoints1);
    printf("done\n");
    
    // extract keypoints from the second image
    vector<KeyPoint> keypoints2;
    surf_extractor(img2, Mat(), keypoints2);
    printf("Extracted %d keypoints from the second image...\n", (int)keypoints2.size());

    
    printf("Finding nearest neighbors...");
    // find NN for each of keypoints2 in keypoints1
    vector<int> desc_idx;
    desc_idx.resize(keypoints2.size());
    for(size_t i = 0; i < keypoints2.size(); i++)
    {
        int pose_idx = 0;
        float distance = 0;
        descriptors.FindDescriptor(img2, keypoints2[i].pt, desc_idx[i], pose_idx, distance);
    }
    printf("done\n");
    
    IplImage* img_corr = DrawCorrespondences(img1, keypoints1, img2, keypoints2, desc_idx);
    
    cvNamedWindow("correspondences", 1);
    cvShowImage("correspondences", img_corr);
    cvWaitKey(0);
    
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&img_corr);
}

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2, const vector<KeyPoint>& features2, const vector<int>& desc_idx)
{
    IplImage* img_corr = cvCreateImage(cvSize(img1->width + img2->width, MAX(img1->height, img2->height)), IPL_DEPTH_8U, 3);
    cvSetImageROI(img_corr, cvRect(0, 0, img1->width, img1->height));
    cvCvtColor(img1, img_corr, CV_GRAY2RGB);
    cvSetImageROI(img_corr, cvRect(img1->width, 0, img2->width, img2->height));
    cvCvtColor(img2, img_corr, CV_GRAY2RGB);
    cvResetImageROI(img_corr);
    
    for(size_t i = 0; i < features1.size(); i++)
    {
        cvCircle(img_corr, features1[i].pt, 3, CV_RGB(255, 0, 0));
    }
    
    for(size_t i = 0; i < features2.size(); i++)
    {
        CvPoint pt = cvPoint(features2[i].pt.x + img1->width, features2[i].pt.y);
        cvCircle(img_corr, pt, 3, CV_RGB(255, 0, 0));
        cvLine(img_corr, features1[desc_idx[i]].pt, pt, CV_RGB(0, 255, 0));
    }
    
    return img_corr;
}

/*
 *  pca_features
 * 
 *
 */

void savePCAFeatures(const char* filename, CvMat* avg, CvMat* eigenvectors)
{
    CvMemStorage* storage = cvCreateMemStorage();
    
    CvFileStorage* fs = cvOpenFileStorage(filename, storage, CV_STORAGE_WRITE);
    cvWrite(fs, "avg", avg);
    cvWrite(fs, "eigenvectors", eigenvectors);
    cvReleaseFileStorage(&fs);   
    
    cvReleaseMemStorage(&storage);
}

void calcPCAFeatures(vector<IplImage*>& patches, const char* filename, CvMat** avg, CvMat** eigenvectors)
{
    int width = patches[0]->width;
    int height = patches[0]->height;
    int length = width*height;
    int patch_count = (int)patches.size();
    
    CvMat* data = cvCreateMat(patch_count, length, CV_32FC1);
    *avg = cvCreateMat(1, length, CV_32FC1);
    CvMat* eigenvalues = cvCreateMat(1, length, CV_32FC1);
    *eigenvectors = cvCreateMat(length, length, CV_32FC1);
    
    for(int i = 0; i < patch_count; i++)
    {
        float sum = cvSum(patches[i]).val[0];
        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                *((float*)(data->data.ptr + data->step*i) + y*width + x) = (float)(unsigned char)patches[i]->imageData[y*patches[i]->widthStep + x]/sum;
            }
        }
    }
    
    printf("Calculating PCA...");
    cvCalcPCA(data, *avg, eigenvalues, *eigenvectors, CV_PCA_DATA_AS_ROW);
    printf("done\n");
    
    // save pca data
    savePCAFeatures(filename, *avg, *eigenvectors);
    
    cvReleaseMat(&data);
    cvReleaseMat(&eigenvalues);
}


void loadPCAFeatures(const char* path, vector<IplImage*>& patches, CvSize patch_size)
{
    const int file_count = 2;
    for(int i = 0; i < file_count; i++)
    {
        char buf[1024];
        sprintf(buf, "%s/one_way_train_%04d.jpg", path, i);
        printf("Reading image %s...", buf);
        IplImage* img = cvLoadImage(buf, CV_LOAD_IMAGE_GRAYSCALE);
        printf("done\n");
        
        vector<KeyPoint> features;
        SURF surf_extractor(1.0f);
        printf("Extracting SURF features...");
        surf_extractor(img, Mat(), features);
        printf("done\n");
        
        for(int j = 0; j < (int)features.size(); j++)
        {
            int patch_width = patch_size.width;
            int patch_height = patch_size.height;
            
            CvPoint center = features[j].pt;
            
            CvRect roi = cvRect(center.x - patch_width/2, center.y - patch_height/2, patch_width, patch_height);
            cvSetImageROI(img, roi);
            roi = cvGetImageROI(img);
            if(roi.width != patch_width || roi.height != patch_height)
            {
                continue;
            }
            
            IplImage* patch = cvCreateImage(cvSize(patch_width, patch_height), IPL_DEPTH_8U, 1);
            cvCopy(img, patch);
            patches.push_back(patch);
            cvResetImageROI(img);
            
        }
        
        printf("Completed file %d, extracted %d features\n", i, (int)features.size());
        
        cvReleaseImage(&img);
    }
}

void generatePCAFeatures(const char* img_filename, const char* pca_filename, CvSize patch_size, CvMat** avg, CvMat** eigenvectors)
{
    vector<IplImage*> patches;
    loadPCAFeatures(img_filename, patches, patch_size);
    calcPCAFeatures(patches, pca_filename, avg, eigenvectors);
}

void generatePCADescriptors(const char* img_path, const char* pca_low_filename, const char* pca_high_filename, 
                            const char* pca_desc_filename, CvSize patch_size)
{
    CvMat* avg_hr;
    CvMat* eigenvectors_hr;
    generatePCAFeatures(img_path, pca_high_filename, patch_size, &avg_hr, &eigenvectors_hr);

    CvMat* avg_lr;
    CvMat* eigenvectors_lr;
    generatePCAFeatures(img_path, pca_low_filename, cvSize(patch_size.width/2, patch_size.height/2), 
        &avg_lr, &eigenvectors_lr);
    
    const int pose_count = 500;
    OneWayDescriptorBase descriptors(patch_size, pose_count);
    descriptors.SetPCAHigh(avg_hr, eigenvectors_hr);
    descriptors.SetPCALow(avg_lr, eigenvectors_lr);
    
    printf("Calculating %d PCA descriptors (you can grab a coffee, this will take a while)...\n", descriptors.GetPCADimHigh());
    descriptors.InitializePoseTransforms();
    descriptors.CreatePCADescriptors();
    descriptors.SavePCADescriptors(pca_desc_filename);
    
    cvReleaseMat(&avg_hr);
    cvReleaseMat(&eigenvectors_hr);
    cvReleaseMat(&avg_lr);
    cvReleaseMat(&eigenvectors_lr);
}
