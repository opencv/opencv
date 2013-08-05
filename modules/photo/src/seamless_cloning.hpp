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
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>
#include <complex>
#include "math.h"

using namespace std;
using namespace cv;

#define pi 3.1416

class Cloning
{

    public:

        Mat grx,gry,sgx,sgy,r_channel,g_channel,b_channel,smask1,grx32,gry32;
        Mat smask,srx32,sry32;
        Mat rx_channel,ry_channel,gx_channel,gy_channel,bx_channel,by_channel,resultr,resultg,resultb;
        void init(Mat &I, Mat &wmask);
        void calc(Mat &I, Mat &gx, Mat &gy, Mat &sx, Mat &sy);
        void getGradientx(const Mat &img, Mat &gx);
        void getGradienty(const Mat &img, Mat &gy);
        void lapx(const Mat &img, Mat &gxx);
        void lapy(const Mat &img, Mat &gyy);
        void dst(double *gtest, double *gfinal,int h,int w);
        void idst(double *gtest, double *gfinal,int h,int w);
        void transpose(double *mat, double *mat_t,int h,int w);
        void poisson_solver(const Mat &img, Mat &gxx , Mat &gyy, Mat &result);
        void normal_clone(Mat &I, Mat &mask, Mat &wmask, Mat &final, int num);
        void local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &final, float red, float green, float blue);
        void illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &final, float alpha, float beta);
        void texture_flatten(Mat &I, Mat &final);
};

void Cloning::getGradientx( const Mat &img, Mat &gx)
{
    int w = img.size().width;
    int h = img.size().height;
    int channel = img.channels();

    gx = Mat::zeros(img.size(),CV_32FC3);
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

    gy = Mat::zeros(img.size(),CV_32FC3);
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

    gxx = Mat::zeros(img.size(),CV_32FC3);
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
    gyy = Mat::zeros(img.size(),CV_32FC3);
    for(int i=0;i<h-1;i++)
        for(int j=0;j<w;j++)
            for(int c=0;c<channel;++c)
            {
                gyy.at<float>(i+1,j*channel+c) =
                    (float)img.at<float>((i+1),j*channel+c) - (float)img.at<float>(i,j*channel+c);

            }
}

void Cloning::dst(double *gtest, double *gfinal,int h,int w)
{

    unsigned long int idx;

    Mat temp = Mat(2*h+2,1,CV_32F);
    Mat res  = Mat(h,1,CV_32F);

    Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

    Mat complex1;
    int p=0;
    for(int i=0;i<w;i++)
    {
        temp.at<float>(0,0) = 0.0;

        for(int j=0,r=1;j<h;j++,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = gtest[idx];
        }

        temp.at<float>(h+1,0)=0.0;

        for(int j=h-1, r=h+2;j>=0;j--,r++)
        {
            idx = j*w+i;
            temp.at<float>(r,0) = -1*gtest[idx];
        }

        merge(planes, 2, complex1);

        dft(complex1,complex1,0,0);

        Mat planes1[] = {Mat::zeros(complex1.size(), CV_32F), Mat::zeros(complex1.size(), CV_32F)};

        split(complex1, planes1); 

        std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

        double fac = -2*imag(two_i);

        for(int c=1,z=0;c<h+1;c++,z++)
        {
            res.at<float>(z,0) = planes1[1].at<float>(c,0)/fac;
        }

        for(int q=0,z=0;q<h;q++,z++)
        {
            idx = q*w+p;
            gfinal[idx] =  res.at<float>(z,0);
        }
        p++;
    }

    temp.release();
    res.release();
    planes[0].release();
    planes[1].release();

}

void Cloning::idst(double *gtest, double *gfinal,int h,int w)
{
    int nn = h+1;
    unsigned long int idx;
    dst(gtest,gfinal,h,w);
    for(int  i= 0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            idx = i*w + j;
            gfinal[idx] = (double) (2*gfinal[idx])/nn;
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
            tmp.at<float>(i,j) = mat[idx];
        }
    }
    Mat tmp_t = tmp.t();

    for(int i = 0;i < tmp_t.size().height; i++)
        for(int j=0;j<tmp_t.size().width;j++)
        {
            idx = i*tmp_t.size().width + j;
            mat_t[idx] = tmp_t.at<float>(i,j);
        }

    tmp.release();

}
void Cloning::poisson_solver(const Mat &img, Mat &gxx , Mat &gyy, Mat &result)
{

    int w = img.size().width;
    int h = img.size().height;

    unsigned long int idx,idx1;

    Mat lap = Mat(img.size(),CV_32FC1);

    for(int i =0;i<h;i++)
        for(int j=0;j<w;j++)
            lap.at<float>(i,j)=gyy.at<float>(i,j)+gxx.at<float>(i,j);

    Mat bound = img.clone();

    for(int i =1;i<h-1;i++)
        for(int j=1;j<w-1;j++)
        {
            bound.at<uchar>(i,j) = 0.0;
        }

    double *f_bp = new double[h*w];


    for(int i =1;i<h-1;i++)
        for(int j=1;j<w-1;j++)
        {
            idx=i*w + j;
            f_bp[idx] = -4*(int)bound.at<uchar>(i,j) + (int)bound.at<uchar>(i,(j+1)) + (int)bound.at<uchar>(i,(j-1))
                + (int)bound.at<uchar>(i-1,j) + (int)bound.at<uchar>(i+1,j);
        }


    Mat diff = Mat(h,w,CV_32FC1);
    for(int i =0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            idx = i*w+j;
            diff.at<float>(i,j) = (lap.at<float>(i,j) - f_bp[idx]);
        }
    }

    lap.release();

    double *gtest = new double[(h-2)*(w-2)];
    for(int i = 0 ; i < h-2;i++)
    {
        for(int j = 0 ; j < w-2; j++)
        {
            idx = i*(w-2) + j;
            gtest[idx] = diff.at<float>(i+1,j+1);

        }
    }

    diff.release();
    ///////////////////////////////////////////////////// Find DST  /////////////////////////////////////////////////////

    double *gfinal = new double[(h-2)*(w-2)];
    double *gfinal_t = new double[(h-2)*(w-2)];
    double *denom = new double[(h-2)*(w-2)];
    double *f3 = new double[(h-2)*(w-2)];
    double *f3_t = new double[(h-2)*(w-2)];
    double *img_d = new double[(h)*(w)];

    dst(gtest,gfinal,h-2,w-2);

    transpose(gfinal,gfinal_t,h-2,w-2);

    dst(gfinal_t,gfinal,w-2,h-2);

    transpose(gfinal,gfinal_t,w-2,h-2);

    int cy=1;

    for(int i = 0 ; i < w-2;i++,cy++)
    {
        for(int j = 0,cx = 1; j < h-2; j++,cx++)
        {
            idx = j*(w-2) + i;
            denom[idx] = (float) 2*cos(pi*cy/( (double) (w-1))) - 2 + 2*cos(pi*cx/((double) (h-1))) - 2;

        }
    }

    for(idx = 0 ; idx < (unsigned)(w-2)*(h-2) ;idx++)
    {
        gfinal_t[idx] = gfinal_t[idx]/denom[idx];
    }


    idst(gfinal_t,f3,h-2,w-2);

    transpose(f3,f3_t,h-2,w-2);

    idst(f3_t,f3,w-2,h-2);

    transpose(f3,f3_t,w-2,h-2);

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
            img_d[idx] = f3_t[idx1];	
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
                result.at<uchar>(i,j) = 255.0;
            else
                result.at<uchar>(i,j) = img_d[idx];	
        }
    }

    delete [] gfinal;
    delete [] gfinal_t;
    delete [] denom;
    delete [] f3;
    delete [] f3_t;
    delete [] img_d;
    delete [] gtest;
    delete [] f_bp;

}

void Cloning::init(Mat &I, Mat &wmask)
{

    grx = Mat(I.size(),CV_32FC3);
    gry = Mat(I.size(),CV_32FC3);
    sgx = Mat(I.size(),CV_32FC3);
    sgy = Mat(I.size(),CV_32FC3);

    r_channel = Mat::zeros(I.size(),CV_8UC1);
    g_channel = Mat::zeros(I.size(),CV_8UC1);
    b_channel = Mat::zeros(I.size(),CV_8UC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            r_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+0); 
            g_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+1); 
            b_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+2);
        }

    smask = Mat(wmask.size(),CV_32FC1);
    srx32 = Mat(I.size(),CV_32FC3);
    sry32 = Mat(I.size(),CV_32FC3);
    smask1 = Mat(wmask.size(),CV_32FC1);
    grx32 = Mat(I.size(),CV_32FC3);
    gry32 = Mat(I.size(),CV_32FC3);
}

void Cloning::calc(Mat &I, Mat &gx, Mat &gy, Mat &sx, Mat &sy)
{

    int channel = I.channels();
    Mat fx = Mat(I.size(),CV_32FC3);
    Mat fy = Mat(I.size(),CV_32FC3);

    for(int i=0;i < I.size().height; i++)
        for(int j=0; j < I.size().width; j++)
            for(int c=0;c<channel;++c)
            {
                fx.at<float>(i,j*channel+c) =
                    (gx.at<float>(i,j*channel+c)+sx.at<float>(i,j*channel+c));
                fy.at<float>(i,j*channel+c) =
                    (gy.at<float>(i,j*channel+c)+sy.at<float>(i,j*channel+c));
            }

    Mat gxx = Mat(I.size(),CV_32FC3);
    Mat gyy = Mat(I.size(),CV_32FC3);

    lapx(fx,gxx);
    lapy(fy,gyy);

    rx_channel = Mat(I.size(),CV_32FC1);
    gx_channel = Mat(I.size(),CV_32FC1);
    bx_channel = Mat(I.size(),CV_32FC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            rx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+0); 
            gx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+1); 
            bx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+2);
        }

    ry_channel = Mat(I.size(),CV_32FC1);
    gy_channel = Mat(I.size(),CV_32FC1);
    by_channel = Mat(I.size(),CV_32FC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            ry_channel.at<float>(i,j) = gyy.at<float>(i,j*3+0); 
            gy_channel.at<float>(i,j) = gyy.at<float>(i,j*3+1); 
            by_channel.at<float>(i,j) = gyy.at<float>(i,j*3+2);
        }

    resultr = Mat(I.size(),CV_8UC1);
    resultg = Mat(I.size(),CV_8UC1);
    resultb = Mat(I.size(),CV_8UC1);

    clock_t tic = clock();


    poisson_solver(r_channel,rx_channel, ry_channel,resultr);
    poisson_solver(g_channel,gx_channel, gy_channel,resultg);
    poisson_solver(b_channel,bx_channel, by_channel,resultb);

    clock_t toc = clock();

    printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);


}
void Cloning::normal_clone(Mat &I, Mat &mask, Mat &wmask, Mat &final, int num)
{
    init(I,wmask);

    int w = I.size().width;
    int h = I.size().height;
    int channel = I.channels();

    getGradientx(I,grx);
    getGradienty(I,gry);

    if(num != 3)
    {
        getGradientx(mask,sgx);
        getGradienty(mask,sgy);
    }

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));

    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);

    wmask.convertTo(smask,CV_32FC1,1.0/255.0);
    I.convertTo(srx32,CV_32FC3,1.0/255.0);
    I.convertTo(sry32,CV_32FC3,1.0/255.0);

    if(num == 1)
    {
        for(int i=0;i < h; i++)
            for(int j=0; j < w; j++)
                for(int c=0;c<channel;++c)
                {
                    srx32.at<float>(i,j*channel+c) =
                        (sgx.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                    sry32.at<float>(i,j*channel+c) =
                        (sgy.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                }

    }
    else if(num == 2)
    {
        for(int i=0;i < h; i++)
            for(int j=0; j < w; j++)
                for(int c=0;c<channel;++c)
                {
                    if(abs(sgx.at<float>(i,j*channel+c) - sgy.at<float>(i,j*channel+c)) >
                            abs(grx.at<float>(i,j*channel+c) - gry.at<float>(i,j*channel+c)))
                    {

                        srx32.at<float>(i,j*channel+c) = sgx.at<float>(i,j*channel+c)
                            * smask.at<float>(i,j);
                        sry32.at<float>(i,j*channel+c) = sgy.at<float>(i,j*channel+c)
                            * smask.at<float>(i,j);
                    }
                    else
                    {
                        srx32.at<float>(i,j*channel+c) = grx.at<float>(i,j*channel+c)
                            * smask.at<float>(i,j);
                        sry32.at<float>(i,j*channel+c) = gry.at<float>(i,j*channel+c)
                            * smask.at<float>(i,j);
                    }
                }
    }
    else if(num == 3)
    {
        Mat gray = Mat(mask.size(),CV_8UC1);
        Mat gray8 = Mat(mask.size(),CV_8UC3);
        cvtColor(mask, gray, COLOR_BGR2GRAY );

        for(int i=0;i<mask.size().height;i++)
            for(int j=0;j<mask.size().width;j++)
            {
                gray8.at<uchar>(i,j*3+0) = gray.at<uchar>(i,j); 
                gray8.at<uchar>(i,j*3+1) = gray.at<uchar>(i,j); 
                gray8.at<uchar>(i,j*3+2) = gray.at<uchar>(i,j);
            }


        getGradientx(gray8,sgx);
        getGradienty(gray8,sgy);

        for(int i=0;i < h; i++)
            for(int j=0; j < w; j++)
                for(int c=0;c<channel;++c)
                {
                    srx32.at<float>(i,j*channel+c) =
                        (sgx.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                    sry32.at<float>(i,j*channel+c) =
                        (sgy.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                }

    }

    bitwise_not(wmask,wmask);

    wmask.convertTo(smask1,CV_32FC1,1.0/255.0);
    I.convertTo(grx32,CV_32FC3,1.0/255.0);
    I.convertTo(gry32,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                grx32.at<float>(i,j*channel+c) =
                    (grx.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
                gry32.at<float>(i,j*channel+c) =
                    (gry.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
            }

    calc(I,grx32,gry32,srx32,sry32);

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            final.at<uchar>(i,j*3+0) = resultr.at<uchar>(i,j);
            final.at<uchar>(i,j*3+1) = resultg.at<uchar>(i,j);
            final.at<uchar>(i,j*3+2) = resultb.at<uchar>(i,j);
        }

}

void Cloning::local_color_change(Mat &I, Mat &mask, Mat &wmask, Mat &final, float red=1.0, float green=1.0, float blue=1.0)
{
    init(I,wmask);

    int w = I.size().width;
    int h = I.size().height;
    int channel = I.channels();

    getGradientx(I,grx);
    getGradienty(I,gry);

    getGradientx(mask,sgx);
    getGradienty(mask,sgy);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));

    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);

    wmask.convertTo(smask,CV_32FC1,1.0/255.0);
    I.convertTo(srx32,CV_32FC3,1.0/255.0);
    I.convertTo(sry32,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                srx32.at<float>(i,j*channel+c) =
                    (sgx.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                sry32.at<float>(i,j*channel+c) =
                    (sgy.at<float>(i,j*channel+c)*smask.at<float>(i,j));
            }

    Mat factor = Mat(I.size(),CV_32FC3);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
        {
            factor.at<float>(i,j*channel+0) = blue;
            factor.at<float>(i,j*channel+1) = green;
            factor.at<float>(i,j*channel+2) = red;
        }



    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                srx32.at<float>(i,j*channel+c) =
                    srx32.at<float>(i,j*channel+c)*factor.at<float>(i,j*channel+c);
                sry32.at<float>(i,j*channel+c) =
                    sry32.at<float>(i,j*channel+c)*factor.at<float>(i,j*channel+c);
            }

    bitwise_not(wmask,wmask);

    wmask.convertTo(smask1,CV_32FC1,1.0/255.0);
    I.convertTo(grx32,CV_32FC3,1.0/255.0);
    I.convertTo(gry32,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                grx32.at<float>(i,j*channel+c) =
                    (grx.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
                gry32.at<float>(i,j*channel+c) =
                    (gry.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
            }

    calc(I,grx32,gry32,srx32,sry32);

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            final.at<uchar>(i,j*3+0) = resultr.at<uchar>(i,j);
            final.at<uchar>(i,j*3+1) = resultg.at<uchar>(i,j);
            final.at<uchar>(i,j*3+2) = resultb.at<uchar>(i,j);
        }

}

void Cloning::illum_change(Mat &I, Mat &mask, Mat &wmask, Mat &final, float alpha, float beta)
{
    init(I,wmask);

    int w = I.size().width;
    int h = I.size().height;
    int channel = I.channels();

    getGradientx(I,grx);
    getGradienty(I,gry);

    getGradientx(mask,sgx);
    getGradienty(mask,sgy);

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));

    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);
    erode(wmask, wmask, Kernel);

    wmask.convertTo(smask,CV_32FC1,1.0/255.0);
    I.convertTo(srx32,CV_32FC3,1.0/255.0);
    I.convertTo(sry32,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                srx32.at<float>(i,j*channel+c) =
                    (sgx.at<float>(i,j*channel+c)*smask.at<float>(i,j));
                sry32.at<float>(i,j*channel+c) =
                    (sgy.at<float>(i,j*channel+c)*smask.at<float>(i,j));
            }


    Mat mag = Mat(I.size(),CV_32FC3);
    I.convertTo(mag,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {

                mag.at<float>(i,j*channel+c) =
                    sqrt(pow(srx32.at<float>(i,j*channel+c),2) + pow(sry32.at<float>(i,j*channel+c),2));
            }

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                if(srx32.at<float>(i,j*channel+c) != 0)
                {
                    srx32.at<float>(i,j*channel+c) =
                        pow(alpha,beta)*srx32.at<float>(i,j*channel+c)*pow(mag.at<float>(i,j*channel+c),-1*beta);
                    sry32.at<float>(i,j*channel+c) =
                        pow(alpha,beta)*sry32.at<float>(i,j*channel+c)*pow(mag.at<float>(i,j*channel+c),-1*beta);
                }
            }

    bitwise_not(wmask,wmask);

    wmask.convertTo(smask1,CV_32FC1,1.0/255.0);
    I.convertTo(grx32,CV_32FC3,1.0/255.0);
    I.convertTo(gry32,CV_32FC3,1.0/255.0);

    for(int i=0;i < h; i++)
        for(int j=0; j < w; j++)
            for(int c=0;c<channel;++c)
            {
                grx32.at<float>(i,j*channel+c) =
                    (grx.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
                gry32.at<float>(i,j*channel+c) =
                    (gry.at<float>(i,j*channel+c)*smask1.at<float>(i,j));
            }

    calc(I,grx32,gry32,srx32,sry32);

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
        {
            final.at<uchar>(i,j*3+0) = resultr.at<uchar>(i,j);
            final.at<uchar>(i,j*3+1) = resultg.at<uchar>(i,j);
            final.at<uchar>(i,j*3+2) = resultb.at<uchar>(i,j);
        }
}

void Cloning::texture_flatten(Mat &I, Mat &final)
{

    grx = Mat(I.size(),CV_32FC3);
    gry = Mat(I.size(),CV_32FC3);

    Mat out = Mat(I.size(),CV_8UC1);

    getGradientx( I, grx);
    getGradienty( I, gry);

    Canny( I, out, 30, 45, 3 );

    int channel = I.channels();

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
            for(int c=0;c<channel;c++)
            {
                if(out.at<uchar>(i,j) != 255)
                {
                    grx.at<float>(i,j*channel+c) = 0.0;
                    gry.at<float>(i,j*channel+c) = 0.0;
                }
            }

    r_channel = Mat::zeros(I.size(),CV_8UC1);
    g_channel = Mat::zeros(I.size(),CV_8UC1);
    b_channel = Mat::zeros(I.size(),CV_8UC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            r_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+0); 
            g_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+1); 
            b_channel.at<uchar>(i,j) = I.at<uchar>(i,j*3+2);
        }

    Mat gxx = Mat(I.size(),CV_32FC3);
    Mat gyy = Mat(I.size(),CV_32FC3);

    lapx(grx,gxx);
    lapy(gry,gyy);

    rx_channel = Mat(I.size(),CV_32FC1);
    gx_channel = Mat(I.size(),CV_32FC1);
    bx_channel = Mat(I.size(),CV_32FC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            rx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+0); 
            gx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+1); 
            bx_channel.at<float>(i,j) = gxx.at<float>(i,j*3+2);
        }

    ry_channel = Mat(I.size(),CV_32FC1);
    gy_channel = Mat(I.size(),CV_32FC1);
    by_channel = Mat(I.size(),CV_32FC1);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            ry_channel.at<float>(i,j) = gyy.at<float>(i,j*3+0); 
            gy_channel.at<float>(i,j) = gyy.at<float>(i,j*3+1); 
            by_channel.at<float>(i,j) = gyy.at<float>(i,j*3+2);
        }

    resultr = Mat(I.size(),CV_8UC1);
    resultg = Mat(I.size(),CV_8UC1);
    resultb = Mat(I.size(),CV_8UC1);

    clock_t tic = clock();


    poisson_solver(r_channel,rx_channel, ry_channel,resultr);
    poisson_solver(g_channel,gx_channel, gy_channel,resultg);
    poisson_solver(b_channel,bx_channel, by_channel,resultb);

    clock_t toc = clock();

    printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    for(int i=0;i<I.size().height;i++)
        for(int j=0;j<I.size().width;j++)
        {
            final.at<uchar>(i,j*3+0) = resultr.at<uchar>(i,j);
            final.at<uchar>(i,j*3+1) = resultg.at<uchar>(i,j);
            final.at<uchar>(i,j*3+2) = resultb.at<uchar>(i,j);
        }
}
