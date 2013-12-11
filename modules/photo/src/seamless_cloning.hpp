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

#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <stdlib.h>
#include <complex>
#include "math.h"

using namespace std;
using namespace cv;

class Cloning
{

    public:

        vector <Mat> rgb_channel, rgbx_channel, rgby_channel, output;
        Mat grx, gry, sgx, sgy, srx32, sry32, grx32, gry32, smask, smask1;
        void init_var(Mat &I, Mat &wmask);
        void initialization(Mat &I, Mat &mask, Mat &wmask);
        void scalar_product(Mat mat, float r, float g, float b);
        void array_product(Mat mat1, Mat mat2, Mat mat3);
        void poisson(Mat &I, Mat &gx, Mat &gy, Mat &sx, Mat &sy);
        void evaluate(Mat &I, Mat &wmask, Mat &cloned);
        void getGradientx(const Mat &img, Mat &gx);
        void getGradienty(const Mat &img, Mat &gy);
        void lapx(const Mat &img, Mat &gxx);
        void lapy(const Mat &img, Mat &gyy);
        void dst(double *mod_diff, double *sineTransform,int h,int w);
        void idst(double *mod_diff, double *sineTransform,int h,int w);
        void transpose(double *mat, double *mat_t,int h,int w);
        void solve(const Mat &img, double *mod_diff, Mat &result);
        void poisson_solver(const Mat &img, Mat &gxx , Mat &gyy, Mat &result);
        void normal_clone(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, int num);
        void local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul, float green_mul, float blue_mul);
        void illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta);
        void texture_flatten(Mat &I, Mat &mask, Mat &wmask, double low_threshold, double high_threhold, int kernel_size, Mat &cloned);
};

void Cloning::getGradientx( const Mat &img, Mat &gx)
{
    int w = img.size().width;
    int h = img.size().height;
    int channel = img.channels();
    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gx.at<float>(i,j*channel+c) =
                    (float)img.at<uchar>(i,(j+1)*channel+c) - (float)img.at<uchar>(i,j*channel+c);
            }

}

void Cloning::getGradienty( const Mat &img, Mat &gy)
{
    int w = img.size().width;
    int h = img.size().height;
    int channel = img.channels();
    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gy.at<float>(i,j*channel+c) =
                    (float)img.at<uchar>((i+1),j*channel+c) - (float)img.at<uchar>(i,j*channel+c);

            }
}

void Cloning::lapx( const Mat &img, Mat &gxx)
{
    int w = img.size().width;
    int h = img.size().height;
    int channel = img.channels();
    for(int i=0;i<h;i++)
        for(int j=0;j<w-1;j++)
            for(int c=0;c<channel;++c)
            {
                gxx.at<float>(i,(j+1)*channel+c) =
                    (float)img.at<float>(i,(j+1)*channel+c) - (float)img.at<float>(i,j*channel+c);
            }
}

void Cloning::lapy( const Mat &img, Mat &gyy)
{
    int w = img.size().width;
    int h = img.size().height;
    int channel = img.channels();
    for(int i=0;i<h-1;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gyy.at<float>(i+1,j*channel+c) =
                    (float)img.at<float>((i+1),j*channel+c) - (float)img.at<float>(i,j*channel+c);

            }
}

void Cloning::dst(double *mod_diff, double *sineTransform,int h,int w)
{

    unsigned long int idx;

    Mat temp = Mat(2*h+2,1,CV_32F);
    Mat res  = Mat(h,1,CV_32F);

    Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

    Mat result;
    int p=0;
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

        std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

        double factor = -2*imag(two_i);

        for(int c=1,z=0;c<h+1;c++,z++)
        {
            res.at<float>(z,0) = (float) (planes1[1].at<float>(c,0)/factor);
        }

        for(int q=0,z=0;q<h;q++,z++)
        {
            idx = q*w+p;
            sineTransform[idx] =  res.at<float>(z,0);
        }
        p++;
    }
}

void Cloning::idst(double *mod_diff, double *sineTransform,int h,int w)
{
    int nn = h+1;
    unsigned long int idx;
    dst(mod_diff,sineTransform,h,w);
    for(int  i= 0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            idx = i*w + j;
            sineTransform[idx] = (double) (2*sineTransform[idx])/nn;
        }

}

void Cloning::transpose(double *mat, double *mat_t,int h,int w)
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

void Cloning::solve(const Mat &img, double *mod_diff, Mat &result)
{
    int w = img.size().width;
    int h = img.size().height;

    unsigned long int idx,idx1;

    double *sineTransform = new double[(h-2)*(w-2)];
    double *sineTransform_t = new double[(h-2)*(w-2)];
    double *denom = new double[(h-2)*(w-2)];
    double *invsineTransform = new double[(h-2)*(w-2)];
    double *invsineTransform_t = new double[(h-2)*(w-2)];
    double *img_d = new double[(h)*(w)];

    dst(mod_diff,sineTransform,h-2,w-2);

    transpose(sineTransform,sineTransform_t,h-2,w-2);

    dst(sineTransform_t,sineTransform,w-2,h-2);

    transpose(sineTransform,sineTransform_t,w-2,h-2);

    int cy = 1;

    for(int i = 0 ; i < w-2;i++,cy++)
    {
        for(int j = 0,cx = 1; j < h-2; j++,cx++)
        {
            idx = j*(w-2) + i;
            denom[idx] = (float) 2*cos(CV_PI*cy/( (double) (w-1))) - 2 + 2*cos(CV_PI*cx/((double) (h-1))) - 2;

        }
    }

    for(idx = 0 ; idx < (unsigned)(w-2)*(h-2) ;idx++)
    {
        sineTransform_t[idx] = sineTransform_t[idx]/denom[idx];
    }

    idst(sineTransform_t,invsineTransform,h-2,w-2);

    transpose(invsineTransform,invsineTransform_t,h-2,w-2);

    idst(invsineTransform_t,invsineTransform,w-2,h-2);

    transpose(invsineTransform,invsineTransform_t,w-2,h-2);

    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            idx = i*w + j;
            img_d[idx] = (double)img.at<uchar>(i,j);
        }
    }
    for(int i = 1 ; i < h-1;i++)
    {
        for(int j = 1 ; j < w-1; j++)
        {
            idx = i*w + j;
            img_d[idx] = 0.0;
        }
    }
    for(int i = 1,id1=0 ; i < h-1;i++,id1++)
    {
        for(int j = 1,id2=0 ; j < w-1; j++,id2++)
        {
            idx = i*w + j;
            idx1= id1*(w-2) + id2;
            img_d[idx] = invsineTransform_t[idx1];
        }
    }

    for(int i = 0 ; i < h;i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            idx = i*w + j;
            if(img_d[idx] < 0.0)
                result.at<uchar>(i,j) = 0;
            else if(img_d[idx] > 255.0)
                result.at<uchar>(i,j) = 255;
            else
                result.at<uchar>(i,j) = (uchar) img_d[idx];
        }
    }

    delete [] sineTransform;
    delete [] sineTransform_t;
    delete [] denom;
    delete [] invsineTransform;
    delete [] invsineTransform_t;
    delete [] img_d;
}

void Cloning::poisson_solver(const Mat &img, Mat &gxx , Mat &gyy, Mat &result)
{

    int w = img.size().width;
    int h = img.size().height;

    unsigned long int idx;

    Mat lap = Mat(img.size(),CV_32FC1);

    lap = gxx + gyy;

    Mat bound = img.clone();

    rectangle(bound, Point(1, 1), Point(img.cols-2, img.rows-2), Scalar::all(0), -1);

    double *boundary_point = new double[h*w];

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

    double *mod_diff = new double[(h-2)*(w-2)];
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

    delete [] mod_diff;
    delete [] boundary_point;
}

void Cloning::init_var(Mat &I, Mat &wmask)
{
    grx = Mat(I.size(),CV_32FC3);
    gry = Mat(I.size(),CV_32FC3);
    sgx = Mat(I.size(),CV_32FC3);
    sgy = Mat(I.size(),CV_32FC3);

    split(I,rgb_channel);

    smask = Mat(wmask.size(),CV_32FC1);
    srx32 = Mat(I.size(),CV_32FC3);
    sry32 = Mat(I.size(),CV_32FC3);
    smask1 = Mat(wmask.size(),CV_32FC1);
    grx32 = Mat(I.size(),CV_32FC3);
    gry32 = Mat(I.size(),CV_32FC3);
}

void Cloning::initialization(Mat &I, Mat &mask, Mat &wmask)
{
    init_var(I,wmask);

    getGradientx(I,grx);
    getGradienty(I,gry);

    getGradientx(mask,sgx);
    getGradienty(mask,sgy);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));

    erode(wmask, wmask, Kernel, Point(-1,-1), 3);

    wmask.convertTo(smask,CV_32FC1,1.0/255.0);
    I.convertTo(srx32,CV_32FC3,1.0/255.0);
    I.convertTo(sry32,CV_32FC3,1.0/255.0);
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

void Cloning::array_product(Mat mat1, Mat mat2, Mat mat3)
{
    vector <Mat> channels_temp1;
    vector <Mat> channels_temp2;
    split(mat1,channels_temp1);
    split(mat2,channels_temp2);
    multiply(channels_temp2[2],mat3,channels_temp1[2]);
    multiply(channels_temp2[1],mat3,channels_temp1[1]);
    multiply(channels_temp2[0],mat3,channels_temp1[0]);
    merge(channels_temp1,mat1);
}

void Cloning::poisson(Mat &I, Mat &gx, Mat &gy, Mat &sx, Mat &sy)
{
    Mat fx = Mat(I.size(),CV_32FC3);
    Mat fy = Mat(I.size(),CV_32FC3);

    fx = gx + sx;
    fy = gy + sy;

    Mat gxx = Mat(I.size(),CV_32FC3);
    Mat gyy = Mat(I.size(),CV_32FC3);

    lapx(fx,gxx);
    lapy(fy,gyy);

    split(gxx,rgbx_channel);
    split(gyy,rgby_channel);

    split(I,output);

    poisson_solver(rgb_channel[2],rgbx_channel[2], rgby_channel[2],output[2]);
    poisson_solver(rgb_channel[1],rgbx_channel[1], rgby_channel[1],output[1]);
    poisson_solver(rgb_channel[0],rgbx_channel[0], rgby_channel[0],output[0]);
}

void Cloning::evaluate(Mat &I, Mat &wmask, Mat &cloned)
{
    bitwise_not(wmask,wmask);

    wmask.convertTo(smask1,CV_32FC1,1.0/255.0);
    I.convertTo(grx32,CV_32FC3,1.0/255.0);
    I.convertTo(gry32,CV_32FC3,1.0/255.0);

    array_product(grx32,grx,smask1);
    array_product(gry32,gry,smask1);

    poisson(I,grx32,gry32,srx32,sry32);

    merge(output,cloned);
}

void Cloning::normal_clone(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, int num)
{
    int w = I.size().width;
    int h = I.size().height;

    initialization(I,mask,wmask);

    if(num == 1)
    {
        array_product(srx32,sgx,smask);
        array_product(sry32,sgy,smask);

    }
    else if(num == 2)
    {
        for(int i=0;i < h; i++)
            for(int j=0; j < w; j++)
            {
                if(abs(sgx.at<float>(i,j) - sgy.at<float>(i,j)) > abs(grx.at<float>(i,j) - gry.at<float>(i,j)))
                {
                    srx32.at<float>(i,j) = sgx.at<float>(i,j) * smask.at<float>(i,j);
                    sry32.at<float>(i,j) = sgy.at<float>(i,j) * smask.at<float>(i,j);
                }
                else
                {
                    srx32.at<float>(i,j) = grx.at<float>(i,j) * smask.at<float>(i,j);
                    sry32.at<float>(i,j) = gry.at<float>(i,j) * smask.at<float>(i,j);
                }
            }
    }
    else if(num == 3)
    {
        Mat gray = Mat(mask.size(),CV_8UC1);
        Mat gray8 = Mat(mask.size(),CV_8UC3);
        cvtColor(mask, gray, COLOR_BGR2GRAY );
        vector <Mat> temp;
        split(I,temp);
        gray.copyTo(temp[2]);
        gray.copyTo(temp[1]);
        gray.copyTo(temp[0]);

        merge(temp,gray8);

        getGradientx(gray8,sgx);
        getGradienty(gray8,sgy);

        array_product(srx32,sgx,smask);
        array_product(sry32,sgy,smask);

    }

    evaluate(I,wmask,cloned);
}

void Cloning::local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul=1.0,
                                 float green_mul=1.0, float blue_mul=1.0)
{
    initialization(I,mask,wmask);

    array_product(srx32,sgx,smask);
    array_product(sry32,sgy,smask);
    scalar_product(srx32,red_mul,green_mul,blue_mul);
    scalar_product(sry32,red_mul,green_mul,blue_mul);

    evaluate(I,wmask,cloned);
}

void Cloning::illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta)
{
    initialization(I,mask,wmask);

    array_product(srx32,sgx,smask);
    array_product(sry32,sgy,smask);

    Mat mag = Mat(I.size(),CV_32FC3);
    magnitude(srx32,sry32,mag);

    Mat multX, multY, multx_temp, multy_temp;

    multiply(srx32,pow(alpha,beta),multX);
    pow(mag,-1*beta, multx_temp);
    multiply(multX,multx_temp,srx32);

    multiply(sry32,pow(alpha,beta),multY);
    pow(mag,-1*beta, multy_temp);
    multiply(multY,multy_temp,sry32);

    Mat zeroMask = (srx32 != 0);

    srx32.copyTo(srx32, zeroMask);
    sry32.copyTo(sry32, zeroMask);

    evaluate(I,wmask,cloned);
}

void Cloning::texture_flatten(Mat &I, Mat &mask, Mat &wmask, double low_threshold,
        double high_threshold, int kernel_size, Mat &cloned)
{
    initialization(I,mask,wmask);

    Mat out = Mat(mask.size(),CV_8UC1);
    Canny(mask,out,low_threshold,high_threshold,kernel_size);

    Mat zeros(sgx.size(), CV_32FC3);
    zeros.setTo(0);
    Mat zerosMask = (out != 255);
    zeros.copyTo(sgx, zerosMask);
    zeros.copyTo(sgy, zerosMask);

    array_product(srx32,sgx,smask);
    array_product(sry32,sgy,smask);

    evaluate(I,wmask,cloned);
}
