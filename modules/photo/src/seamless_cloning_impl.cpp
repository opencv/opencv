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
#include "opencv2/highgui.hpp"

#include <iostream>
#include <complex>

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

void Cloning::dst(const std::vector<float>& mod_diff, std::vector<float>& sineTransform,int h,int w)
{

    unsigned long int idx;

    Mat temp = Mat(2*h+2,1,CV_32F);
    Mat res  = Mat(h,1,CV_32F);

    Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

    Mat result;
    int p=0;

    const float factor = 0.5;

    for(int i=0;i<w;i++)
    {
        temp.at<float>(0,0) = 0.0;

        for(int j=0,r=1;j<h;j++,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = (float) mod_diff[idx];
        }

        temp.at<float>(h+1,0)=0.0;

        for(int j=h-1, r=h+2;j>=0;j--,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = (float) (-1.0 * mod_diff[idx]);
        }

        merge(planes, 2, result);

        dft(result,result,0,0);

        Mat planes1[] = {Mat::zeros(result.size(), CV_32F), Mat::zeros(result.size(), CV_32F)};

        split(result, planes1);

        for(int c=1,z=0;c<h+1;c++,z++)
        {
            res.at<float>(z,0) = (float) (planes1[1].at<float>(c,0) * factor);
        }

        for(int q=0,z=0;q<h;q++,z++)
        {
            idx = q*w+p;
            sineTransform[idx] =  res.at<float>(z,0);
        }
        p++;
    }
}

void Cloning::idst(const std::vector<float>& mod_diff, std::vector<float>& sineTransform,int h,int w)
{
    int nn = h+1;
    unsigned long int idx;
    dst(mod_diff,sineTransform,h,w);
    for(int  i= 0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            idx = i*w + j;
            sineTransform[idx] = (float) (2*sineTransform[idx])/nn;
        }

}

void Cloning::transpose(const std::vector<float>& mat, std::vector<float>& mat_t,int h,int w)
{

    Mat tmp = Mat(h,w,CV_32FC1);
    unsigned long int idx;
    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {

            idx = i*(w) + j;
            tmp.at<float>(i,j) = (float) mat[idx];
        }
    }
    Mat tmp_t = tmp.t();

    for(int i = 0;i < tmp_t.size().height; i++)
        for(int j=0;j<tmp_t.size().width;j++)
        {
            idx = i*tmp_t.size().width + j;
            mat_t[idx] = tmp_t.at<float>(i,j);
        }
}

void Cloning::solve(const Mat &img, const std::vector<float>& mod_diff, Mat &result)
{
    const int w = img.size().width;
    const int h = img.size().height;

    std::vector<float> sineTransform((h-2)*(w-2), 0.);
    std::vector<float> sineTranformTranspose((h-2)*(w-2), 0.);
    std::vector<float> denom((h-2)*(w-2), 0.);
    std::vector<float> invsineTransform((h-2)*(w-2), 0.);
    std::vector<float> invsineTransform_t((h-2)*(w-2), 0.);

    dst(mod_diff,sineTransform,h-2,w-2);

    transpose(sineTransform,sineTranformTranspose,h-2,w-2);

    dst(sineTranformTranspose,sineTransform,w-2,h-2);

    transpose(sineTransform,sineTranformTranspose,w-2,h-2);


    for(int j = 0,cx = 1; j < h-2; j++,cx++)
    {
        for(int i = 0, cy=1 ; i < w-2;i++,cy++)
        {
            int idx = j*(w-2) + i;
            denom[idx] = 2*cos(CV_PI*cy/( w-1)) + 2*cos(CV_PI*cx/(h-1)) - 4;

        }
    }

    for(int idx = 0 ; idx < (w-2)*(h-2) ;idx++)
    {
        sineTranformTranspose[idx] = sineTranformTranspose[idx]/denom[idx];
    }

    idst(sineTranformTranspose,invsineTransform,h-2,w-2);

    transpose(invsineTransform,invsineTransform_t,h-2,w-2);

    idst(invsineTransform_t,invsineTransform,w-2,h-2);

    transpose(invsineTransform,invsineTransform_t,w-2,h-2);

     //first col
    for(int i = 0 ; i < w ; ++i)
        result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

    for(int j = 1 ; j < h-1 ; ++j)
    {
        //first row
        result.ptr<unsigned char>(j)[0] = img.ptr<unsigned char>(j)[0];
        
        for(int i = 1 ; i < w-1 ; ++i)
        {
            int idx = (j-1)* (w-2) + (i-1);
            //saturate cast is not used here, because it behaves differently from the previous implementation
            //most notable, saturate_cast rounds before truncating, here it's the opposite.
            float value = invsineTransform_t[idx];
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

void Cloning::poisson_solver(const Mat &img, Mat &laplacianX , Mat &laplacianY, Mat &result)
{

    const int w = img.size().width;
    const int h = img.size().height;

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

void Cloning::init_var(const Mat &destination, const Mat &binaryMask)
{
    destinationGradientX = Mat(destination.size(),CV_32FC3);
    destinationGradientY = Mat(destination.size(),CV_32FC3);
    patchGradientX = Mat(destination.size(),CV_32FC3);
    patchGradientY = Mat(destination.size(),CV_32FC3);

    binaryMaskFloat = Mat(binaryMask.size(),CV_32FC1);
    binaryMaskFloatInverted = Mat(binaryMask.size(),CV_32FC1);
}

void Cloning::compute_derivatives(const Mat& destination, const Mat &patch, const Mat &binaryMask)
{
    init_var(destination,binaryMask);

    computeGradientX(destination,destinationGradientX);
    computeGradientY(destination,destinationGradientY);

    computeGradientX(patch,patchGradientX);
    computeGradientY(patch,patchGradientY);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));
    erode(binaryMask, binaryMask, Kernel, Point(-1,-1), 3);

    binaryMask.convertTo(binaryMaskFloat,CV_32FC1,1.0/255.0);
}

void Cloning::scalar_product(Mat mat, float r, float g, float b)
{
    vector <Mat> channels;
    split(mat,channels);
    multiply(channels[2],r,channels[2]);
    multiply(channels[1],g,channels[1]);
    multiply(channels[0],b,channels[0]);
    merge(channels,mat);
}

void Cloning::array_product(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
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
        poisson_solver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
    }
}

void Cloning::evaluate(const Mat &I, const Mat &wmask, const Mat &cloned)
{
    bitwise_not(wmask,wmask);

    wmask.convertTo(binaryMaskFloatInverted,CV_32FC1,1.0/255.0);

    array_product(destinationGradientX,binaryMaskFloatInverted, destinationGradientX);
    array_product(destinationGradientY,binaryMaskFloatInverted, destinationGradientY);

    poisson(I);

    merge(output,cloned);
}

void Cloning::normal_clone(const Mat &destination, const Mat &patch, const Mat &binaryMask, Mat &cloned, int flag)
{
    int w = destination.size().width;
    int h = destination.size().height;
    int channel = destination.channels();

    compute_derivatives(destination,patch,binaryMask);

    switch(flag)
    {
        case NORMAL_CLONE:
            array_product(patchGradientX,binaryMaskFloat, patchGradientX);
            array_product(patchGradientY,binaryMaskFloat, patchGradientY);
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

            array_product(patchGradientX, binaryMaskFloat, patchGradientX);
            array_product(patchGradientY, binaryMaskFloat, patchGradientY);
        break;

    }

    evaluate(destination,binaryMask,cloned);
}

void Cloning::local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul=1.0,
                                 float green_mul=1.0, float blue_mul=1.0)
{
    compute_derivatives(I,mask,wmask);

    array_product(patchGradientX,binaryMaskFloat, patchGradientX);
    array_product(patchGradientY,binaryMaskFloat, patchGradientY);
    scalar_product(patchGradientX,red_mul,green_mul,blue_mul);
    scalar_product(patchGradientY,red_mul,green_mul,blue_mul);

    evaluate(I,wmask,cloned);
}

void Cloning::illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta)
{
    compute_derivatives(I,mask,wmask);

    array_product(patchGradientX,binaryMaskFloat, patchGradientX);
    array_product(patchGradientY,binaryMaskFloat, patchGradientY);

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

void Cloning::texture_flatten(Mat &I, Mat &mask, Mat &wmask, float low_threshold,
        float high_threshold, int kernel_size, Mat &cloned)
{
    compute_derivatives(I,mask,wmask);

    Mat out = Mat(mask.size(),CV_8UC1);
    Canny(mask,out,low_threshold,high_threshold,kernel_size);

    Mat zeros(patchGradientX.size(), CV_32FC3);
    zeros.setTo(0);
    Mat zerosMask = (out != 255);
    zeros.copyTo(patchGradientX, zerosMask);
    zeros.copyTo(patchGradientY, zerosMask);

    array_product(patchGradientX,binaryMaskFloat, patchGradientX);
    array_product(patchGradientY,binaryMaskFloat, patchGradientY);

    evaluate(I,wmask,cloned);
}
