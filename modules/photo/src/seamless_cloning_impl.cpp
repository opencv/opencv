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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "seamless_cloning.hpp"

using namespace cv;
using namespace std;


void Cloning::computeGradientX( const Mat &img, Mat &gx)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,2) = 1;
    kernel.at<char>(0,1) = -1;
    filter2D(img, gx, CV_32F, kernel);
}

void Cloning::computeGradientY( const Mat &img, Mat &gy)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(2,0) = 1;
    kernel.at<char>(1,0) = -1;
    filter2D(img, gy, CV_32F, kernel);
}

void Cloning::computeLaplacianX( const Mat &img, Mat &laplacianX)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(0,1) = 1;
    filter2D(img, laplacianX, CV_32F, kernel);
}

void Cloning::computeLaplacianY( const Mat &img, Mat &laplacianY)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(0,0) = -1;
    kernel.at<char>(1,0) = 1;
    filter2D(img, laplacianY, CV_32F, kernel);
}

void Cloning::dst(const Mat& src, Mat& dest, bool invert)
{
    Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

    int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE: DFT_ROWS;
    
    src.copyTo(temp(Rect(1,0, src.cols, src.rows)));

    for(int j = 0 ; j < src.rows ; ++j)
    {
        for(int i = 0 ; i < src.cols ; ++i)
        {
            temp.ptr<float>(j)[src.cols + 2 + i] = - src.ptr<float>(j)[src.cols - 1 - i];
        }
    }

    Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    Mat complex;

    merge(planes, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes);
    
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

    for(int j = 0 ; j < src.cols ; ++j)
    {
        for(int i = 0 ; i < src.rows ; ++i)
        {
            float val = planes[1].ptr<float>(i)[j + 1];
            temp.ptr<float>(j)[i + 1] = val;
            temp.ptr<float>(j)[temp.cols - 1 - i] = - val;
        }
    }
 
    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};

    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes2);

    temp = planes2[1].t();
    dest = Mat::zeros(src.size(), CV_32F);
    temp(Rect( 0, 1, src.cols, src.rows)).copyTo(dest);
}

void Cloning::idst(const Mat& src, Mat& dest)
{
    dst(src, dest, true);
}

void Cloning::solve(const Mat &img, std::vector<float>& mod_diff, Mat &result)
{
    const int w = img.cols;
    const int h = img.rows;


    Mat ModDiff(h-2, w-2, CV_32F, &mod_diff[0]);
    Mat res;
    dst(ModDiff, res);
    
    for(int j = 0 ; j < h-2; j++)
    {
        for(int i = 0 ; i < w-2; i++)
        {
            res.ptr<float>(j)[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }

    idst(res, ModDiff);

     //first col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j)
    {
        //first row
        result.ptr<unsigned char>(j)[0] = img.ptr<unsigned char>(j)[0];
        
        for(int i = 1 ; i < w-1 ; ++i)
        {
            //saturate cast is not used here, because it behaves differently from the previous implementation
            //most notable, saturate_cast rounds before truncating, here it's the opposite.
            float value = ModDiff.ptr<float>(j-1)[i-1];
            if(value < 0.)
                result.ptr<unsigned char>(j)[i] = 0;
            else if (value > 255.0)
                result.ptr<unsigned char>(j)[i] = 255;
            else
                result.ptr<unsigned char>(j)[i] = static_cast<unsigned char>(value);
        }

        //last row
        result.ptr<unsigned char>(j)[w-1] = img.ptr<unsigned char>(j)[w-1];
    }

    //last col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<unsigned char>(h-1)[i] = img.ptr<unsigned char>(h-1)[i];


}

void Cloning::poissonSolver(const Mat &img, Mat &laplacianX , Mat &laplacianY, Mat &result)
{

    const int w = img.cols;
    const int h = img.rows;

    unsigned long int idx;

    Mat lap = Mat(img.size(),CV_32FC1);

    lap = laplacianX + laplacianY;

    Mat bound = img.clone();

    rectangle(bound, Point(1, 1), Point(img.cols-2, img.rows-2), Scalar::all(0), -1);


    std::vector<float> boundary_point(h*w, 0.);

    for(int i =1;i<h-1;i++)
        for(int j=1;j<w-1;j++)
        {
            idx=i*w + j;
            boundary_point[idx] = -4*(int)bound.at<uchar>(i,j) + (int)bound.at<uchar>(i,(j+1)) + (int)bound.at<uchar>(i,(j-1))
                + (int)bound.at<uchar>(i-1,j) + (int)bound.at<uchar>(i+1,j);
        }

    Mat diff = Mat(h,w,CV_32FC1);
    for(int i =0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            idx = i*w+j;
            diff.at<float>(i,j) = (float) (lap.at<float>(i,j) - boundary_point[idx]);
        }
    }

    std::vector<float> mod_diff((h-2)*(w-2), 0.);
    for(int i = 0 ; i < h-2;i++)
    {
        for(int j = 0 ; j < w-2; j++)
        {
            idx = i*(w-2) + j;
            mod_diff[idx] = diff.at<float>(i+1,j+1);

        }
    }
    ///////////////////////////////////////////////////// Find DST  /////////////////////////////////////////////////////

    solve(img,mod_diff,result);
}

void Cloning::initVariables(const Mat &destination, const Mat &binaryMask)
{
    destinationGradientX = Mat(destination.size(),CV_32FC3);
    destinationGradientY = Mat(destination.size(),CV_32FC3);
    patchGradientX = Mat(destination.size(),CV_32FC3);
    patchGradientY = Mat(destination.size(),CV_32FC3);

    binaryMaskFloat = Mat(binaryMask.size(),CV_32FC1);
    binaryMaskFloatInverted = Mat(binaryMask.size(),CV_32FC1);

    //init of the filters used in the dst
    const int w = destination.cols;
    filter_X.resize(w - 2);
    for(int i = 0 ; i < w-2 ; ++i)
        filter_X[i] = 2.0f * std::cos(CV_PI * (i + 1) / (w - 1));

    const int h  = destination.rows;
    filter_Y.resize(h - 2);
    for(int j = 0 ; j < h - 2 ; ++j)
        filter_Y[j] = 2.0f * std::cos(CV_PI * (j + 1) / (h - 1));
}

void Cloning::computeDerivatives(const Mat& destination, const Mat &patch, const Mat &binaryMask)
{
    initVariables(destination,binaryMask);

    computeGradientX(destination,destinationGradientX);
    computeGradientY(destination,destinationGradientY);

    computeGradientX(patch,patchGradientX);
    computeGradientY(patch,patchGradientY);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));
    erode(binaryMask, binaryMask, Kernel, Point(-1,-1), 3);

    binaryMask.convertTo(binaryMaskFloat,CV_32FC1,1.0/255.0);
}

void Cloning::scalarProduct(Mat mat, float r, float g, float b)
{
    vector <Mat> channels;
    split(mat,channels);
    multiply(channels[2],r,channels[2]);
    multiply(channels[1],g,channels[1]);
    multiply(channels[0],b,channels[0]);
    merge(channels,mat);
}

void Cloning::arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
{
    vector <Mat> lhs_channels;
    vector <Mat> result_channels;
    
    split(lhs,lhs_channels);
    split(result,result_channels);
    
    for(int chan = 0 ; chan < 3 ; ++chan)
        multiply(lhs_channels[chan],rhs,result_channels[chan]);
    
    merge(result_channels,result);
}

void Cloning::poisson(const Mat &destination)
{
    Mat laplacianX = Mat(destination.size(),CV_32FC3);
    Mat laplacianY = Mat(destination.size(),CV_32FC3);

    laplacianX = destinationGradientX + patchGradientX;
    laplacianY = destinationGradientY + patchGradientY;

    computeLaplacianX(laplacianX,laplacianX);
    computeLaplacianY(laplacianY,laplacianY);

    split(laplacianX,rgbx_channel);
    split(laplacianY,rgby_channel);

    split(destination,output);

    for(int chan = 0 ; chan < 3 ; ++chan)
    {
        poissonSolver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
    }
}

void Cloning::evaluate(const Mat &I, const Mat &wmask, const Mat &cloned)
{
    bitwise_not(wmask,wmask);

    wmask.convertTo(binaryMaskFloatInverted,CV_32FC1,1.0/255.0);

    arrayProduct(destinationGradientX,binaryMaskFloatInverted, destinationGradientX);
    arrayProduct(destinationGradientY,binaryMaskFloatInverted, destinationGradientY);

    poisson(I);

    merge(output,cloned);
}

void Cloning::normalClone(const Mat &destination, const Mat &patch, const Mat &binaryMask, Mat &cloned, int flag)
{
    int w = destination.cols;
    int h = destination.rows;
    int channel = destination.channels();

    computeDerivatives(destination,patch,binaryMask);

    switch(flag)
    {
        case NORMAL_CLONE:
            arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
            arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);
            break;

        case MIXED_CLONE:

            for(int i=0;i < h; i++)
            {
               for(int j=0; j < w; j++)
                {
                    for(int c=0;c<channel;++c)
                    {
                        if(abs(patchGradientX.at<float>(i,j*channel+c) - patchGradientY.at<float>(i,j*channel+c)) >
                                abs(destinationGradientX.at<float>(i,j*channel+c) - destinationGradientY.at<float>(i,j*channel+c)))
                        {

                            patchGradientX.at<float>(i,j*channel+c) = patchGradientX.at<float>(i,j*channel+c)
                                * binaryMaskFloat.at<float>(i,j);
                            patchGradientY.at<float>(i,j*channel+c) = patchGradientY.at<float>(i,j*channel+c)
                                * binaryMaskFloat.at<float>(i,j);
                        }
                        else
                        {
                            patchGradientX.at<float>(i,j*channel+c) = destinationGradientX.at<float>(i,j*channel+c)
                                * binaryMaskFloat.at<float>(i,j);
                            patchGradientY.at<float>(i,j*channel+c) = destinationGradientY.at<float>(i,j*channel+c)
                                * binaryMaskFloat.at<float>(i,j);
                        }
                    }
                }
            }
            break;

        case MONOCHROME_TRANSFER:
            Mat gray = Mat(patch.size(),CV_8UC1);
            Mat gray8 = Mat(patch.size(),CV_8UC3);
            cvtColor(patch, gray, COLOR_BGR2GRAY );
            vector <Mat> temp;
            split(destination,temp);
            gray.copyTo(temp[2]);
            gray.copyTo(temp[1]);
            gray.copyTo(temp[0]);

            merge(temp,gray8);

            computeGradientX(gray8,patchGradientX);
            computeGradientY(gray8,patchGradientY);

            arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
            arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
        break;

    }

    evaluate(destination,binaryMask,cloned);
}

void Cloning::localColorChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul=1.0,
                                 float green_mul=1.0, float blue_mul=1.0)
{
    computeDerivatives(I,mask,wmask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);
    scalarProduct(patchGradientX,red_mul,green_mul,blue_mul);
    scalarProduct(patchGradientY,red_mul,green_mul,blue_mul);

    evaluate(I,wmask,cloned);
}

void Cloning::illuminationChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta)
{
    computeDerivatives(I,mask,wmask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);

    Mat mag = Mat(I.size(),CV_32FC3);
    magnitude(patchGradientX,patchGradientY,mag);

    Mat multX, multY, multx_temp, multy_temp;

    multiply(patchGradientX,pow(alpha,beta),multX);
    pow(mag,-1*beta, multx_temp);
    multiply(multX,multx_temp, patchGradientX);
    patchNaNs(patchGradientX);

    multiply(patchGradientY,pow(alpha,beta),multY);
    pow(mag,-1*beta, multy_temp);
    multiply(multY,multy_temp,patchGradientY);
    patchNaNs(patchGradientY);

    Mat zeroMask = (patchGradientX != 0);

    patchGradientX.copyTo(patchGradientX, zeroMask);
    patchGradientY.copyTo(patchGradientY, zeroMask);

    evaluate(I,wmask,cloned);
}

void Cloning::textureFlatten(Mat &I, Mat &mask, Mat &wmask, float low_threshold,
        float high_threshold, int kernel_size, Mat &cloned)
{
    computeDerivatives(I,mask,wmask);

    Mat out = Mat(mask.size(),CV_8UC1);
    Canny(mask,out,low_threshold,high_threshold,kernel_size);

    Mat zeros(patchGradientX.size(), CV_32FC3);
    zeros.setTo(0);
    Mat zerosMask = (out != 255);
    zeros.copyTo(patchGradientX, zerosMask);
    zeros.copyTo(patchGradientY, zerosMask);

    arrayProduct(patchGradientX,binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY,binaryMaskFloat, patchGradientY);

    evaluate(I,wmask,cloned);
}
