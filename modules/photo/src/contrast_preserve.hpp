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
#include "math.h"
#include <vector>
#include <limits>

using namespace std;
using namespace cv;

class Decolor
{
    private:
        Mat kernel;
        Mat kernel1;
        int order;

    public:
        void init();
        vector<double> product(vector < vector<int> > &comb, vector <double> &initRGB);
        double energyCalcu(vector <double> &Cg, vector < vector <double> > &polyGrad, vector <double> &wei);
        void singleChannelGradx(const Mat &img, Mat& dest);
        void singleChannelGrady(const Mat &img, Mat& dest);
        void gradvector(const Mat &img, vector <double> &grad);
        void colorGrad(Mat img, vector <double> &Cg);
        void add_vector(vector < vector <int> > &comb, int r,int g,int b);
        void add_to_vector_poly(vector < vector <double> > &polyGrad, vector <double> &curGrad);
        void weak_order(Mat img, vector <double> &alf);
        void grad_system(Mat img, vector < vector < double > > &polyGrad,
                vector < double > &Cg, vector < vector <int> >& comb);
        void wei_update_matrix(vector < vector <double> > &poly, vector <double> &Cg, Mat &X);
        void wei_inti(vector < vector <int> > &comb, vector <double> &wei);
        void grayImContruct(vector <double> &wei, Mat img, Mat &Gray);
};

int rounding(double a);
double power(double term, int p);
double sqroot(double m);

int rounding(double a)
{
    return int(a + 0.5);
}

double power(double term, int p)
{
    double res = 1.0;
    for(int i=0;i<p;i++)
    {
        res *= term;
    }
    return res;
}

double sqroot(double m)
{
    double i=0;
    double x1,x2;
    while( (i*i) <= m )
        i+=0.1;
    x1=i;
    for(int j=0;j<10;j++)
    {
        x2=m;
        x2/=x1;
        x2+=x1;
        x2/=2;
        x1=x2;
    }
    return x2;
}

float sigma = .02;

double Decolor::energyCalcu(vector <double> &Cg, vector < vector <double> > &polyGrad, vector <double> &wei)
{
    vector <double> P;
    vector <double> temp;
    vector <double> temp1;

    double val = 0.0;
    for(unsigned int i=0;i< polyGrad[0].size();i++)
    {
        val = 0.0;
        for(unsigned int j =0;j<polyGrad.size();j++)
            val = val + (polyGrad[j][i] * wei[j]);
        temp.push_back(val - Cg[i]);
        temp1.push_back(val + Cg[i]);
    }

    for(unsigned int i=0;i<polyGrad[0].size();i++)
        P.push_back(-1.0*log(exp(-1.0*power(temp[i],2)/sigma) + exp(-1.0*power(temp1[i],2)/sigma)));

    double sum = 0.0;
    for(unsigned int i=0;i<polyGrad[0].size();i++)
        sum +=P[i];

    return (sum/polyGrad[0].size());

}

void Decolor::init()
{
    kernel = Mat(1,2, CV_32FC1);
    kernel1 = Mat(2,1, CV_32FC1);
    kernel.at<float>(0,0)=1.0;
    kernel.at<float>(0,1)=-1.0;
    kernel1.at<float>(0,0)=1.0;
    kernel1.at<float>(1,0)=-1.0;
    order = 2;
}

vector<double> Decolor::product(vector < vector<int> > &comb, vector <double> &initRGB)
{
    vector <double> res;
    double dp;
    for (unsigned int i=0;i<comb.size();i++)
    {
        dp = 0.0;
        for(int j=0;j<3;j++)
            dp = dp + (comb[i][j] * initRGB[j]);
        res.push_back(dp);
    }
    return res;
}

void Decolor::singleChannelGradx(const Mat &img, Mat& dest)
{
    int w=img.size().width;
    int h=img.size().height;
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    filter2D(img, dest, -1, kernel, anchor, 0.0, BORDER_CONSTANT);
    for(int i=0;i<h;i++)
        dest.at<float>(i,w-1)=0.0;
}

void Decolor::singleChannelGrady(const Mat &img, Mat& dest)
{
    int w=img.size().width;
    int h=img.size().height;
    Point anchor(kernel1.cols - kernel1.cols/2 - 1, kernel1.rows - kernel1.rows/2 - 1);
    filter2D(img, dest, -1, kernel1, anchor, 0.0, BORDER_CONSTANT);
    for(int j=0;j<w;j++)
        dest.at<float>(h-1,j)=0.0;
}

void Decolor::gradvector(const Mat &img, vector <double> &grad)
{
    Mat dest= Mat(img.size().height,img.size().width, CV_32FC1);
    Mat dest1= Mat(img.size().height,img.size().width, CV_32FC1);
    singleChannelGradx(img,dest);
    singleChannelGrady(img,dest1);

    Mat d_trans=dest.t();
    Mat d1_trans=dest1.t();

    int height = d_trans.size().height;
    int width = d_trans.size().width;

    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            grad.push_back(d_trans.at<float>(i,j));

    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            grad.push_back(d1_trans.at<float>(i,j));
    dest.release();
    dest1.release();
}

void Decolor::colorGrad(Mat img, vector <double> &Cg)
{

    Mat lab = Mat(img.size(),CV_32FC3);
    Mat l_channel = Mat(img.size(),CV_32FC1);
    Mat a_channel = Mat(img.size(),CV_32FC1);
    Mat b_channel = Mat(img.size(),CV_32FC1);

    cvtColor(img,lab,COLOR_BGR2Lab);
    for(int i=0;i<img.size().height;i++)
        for(int j=0;j<img.size().width;j++)
        {
            l_channel.at<float>(i,j) = lab.at<float>(i,j*3+0);
            a_channel.at<float>(i,j) = lab.at<float>(i,j*3+1);
            b_channel.at<float>(i,j) = lab.at<float>(i,j*3+2);
        }

    vector <double> ImL;
    vector <double> Ima;
    vector <double> Imb;
    gradvector(l_channel,ImL);
    gradvector(a_channel,Ima);
    gradvector(b_channel,Imb);

    double res =0.0;
    for(unsigned int i=0;i<ImL.size();i++)
    {
        res=sqrt(power(ImL[i],2) + power(Ima[i],2) + power(Imb[i],2))/100;
        Cg.push_back(res);
    }
    lab.release();
    l_channel.release();
    a_channel.release();
    b_channel.release();

    ImL.clear();
    Ima.clear();
    Imb.clear();
}

void Decolor::add_vector(vector < vector <int> > &comb, int r,int g,int b)
{
    static int idx =0;
    comb.push_back( vector <int>() );
    comb.at(idx).push_back( r );
    comb.at(idx).push_back( g );
    comb.at(idx).push_back( b );
    idx++;
}

void Decolor::add_to_vector_poly(vector < vector <double> > &polyGrad, vector <double> &curGrad)
{
    static int idx1 =0;
    polyGrad.push_back( vector <double>() );
    for(unsigned int i=0;i<curGrad.size();i++)
        polyGrad.at(idx1).push_back(curGrad[i]);
    idx1++;
}

void Decolor::weak_order(Mat img, vector <double> &alf)
{
    int h = img.size().height;
    int w = img.size().width;
    double sizefactor;
    if((h + w) > 800)
    {
        sizefactor = (double)800/(h+w);
        resize(img,img,Size(rounding(h*sizefactor),rounding(w*sizefactor)));
    }

    Mat curIm = Mat(img.size(),CV_32FC1);
    Mat red = Mat(img.size(),CV_32FC1);
    Mat green = Mat(img.size(),CV_32FC1);
    Mat blue = Mat(img.size(),CV_32FC1);

    for(int i=0;i<img.size().height;i++)
        for(int j=0;j<img.size().width;j++)
        {
            red.at<float>(i,j) = img.at<float>(i,j*3+2);
            green.at<float>(i,j) = img.at<float>(i,j*3+1);
            blue.at<float>(i,j) = img.at<float>(i,j*3+0);
        }

    vector <double> Rg;
    vector <double> Gg;
    vector <double> Bg;

    vector <double> t1;
    vector <double> t2;
    vector <double> t3;

    vector <double> tmp1;
    vector <double> tmp2;
    vector <double> tmp3;

    gradvector(red,Rg);
    gradvector(green,Gg);
    gradvector(blue,Bg);

    double level = .05;

    for(unsigned int i=0;i<Rg.size();i++)
    {
        if(Rg[i] > level)
            t1.push_back(1.0);
        else
            t1.push_back(0.0);
    }
    for(unsigned int i=0;i<Gg.size();i++)
    {
        if(Gg[i] > level)
            t2.push_back(1.0);
        else
            t2.push_back(0.0);
    }
    for(unsigned int i=0;i<Bg.size();i++)
    {
        if(Bg[i] > level)
            t3.push_back(1.0);
        else
            t3.push_back(0.0);
    }
    for(unsigned int i=0;i<Rg.size();i++)
    {
        if(Rg[i] < -1.0*level)
            tmp1.push_back(1.0);
        else
            tmp1.push_back(0.0);
    }
    for(unsigned int i=0;i<Gg.size();i++)
    {
        if(Gg[i] < -1.0*level)
            tmp2.push_back(1.0);
        else
            tmp2.push_back(0.0);
    }
    for(unsigned int i=0;i<Bg.size();i++)
    {
        if(Bg[i] < -1.0*level)
            tmp3.push_back(1.0);
        else
            tmp3.push_back(0.0);
    }
    for(unsigned int i =0 ;i < Rg.size();i++)
        alf.push_back(t1[i] * t2[i] * t3[i]);

    for(unsigned int i =0 ;i < Rg.size();i++)
        alf[i] -= tmp1[i] * tmp2[i] * tmp3[i];

    double sum =0.0;
    for(unsigned int i=0;i<alf.size();i++)
        sum += abs(alf[i]);

    sum = (double)100*sum/alf.size();

    red.release();
    green.release();
    blue.release();
    curIm.release();

    Rg.clear();
    Gg.clear();
    Bg.clear();

    t1.clear();
    t2.clear();
    t3.clear();

    tmp1.clear();
    tmp2.clear();
    tmp3.clear();
}

void Decolor::grad_system(Mat img, vector < vector < double > > &polyGrad,
        vector < double > &Cg, vector < vector <int> >& comb)
{
    int h = img.size().height;
    int w = img.size().width;

    double sizefactor;
    if((h + w) > 800)
    {
        sizefactor = (double)800/(h+w);
        resize(img,img,Size(rounding(h*sizefactor),rounding(w*sizefactor)));
    }

    h = img.size().height;
    w = img.size().width;
    colorGrad(img,Cg);

    Mat curIm = Mat(img.size(),CV_32FC1);
    Mat red = Mat(img.size(),CV_32FC1);
    Mat green = Mat(img.size(),CV_32FC1);
    Mat blue = Mat(img.size(),CV_32FC1);

    for(int i=0;i<img.size().height;i++)
        for(int j=0;j<img.size().width;j++)
        {
            red.at<float>(i,j) = img.at<float>(i,j*3+2);
            green.at<float>(i,j) = img.at<float>(i,j*3+1);
            blue.at<float>(i,j) = img.at<float>(i,j*3+0);
        }

    for(int r=0 ;r <=order; r++)
        for(int g=0; g<=order;g++)
            for(int b =0; b <=order;b++)
            {
                if((r+g+b)<=order && (r+g+b) > 0)
                {
                    add_vector(comb,r,g,b);
                    for(int i = 0;i<h;i++)
                        for(int j=0;j<w;j++)
                            curIm.at<float>(i,j)=
                                power(red.at<float>(i,j),r)*power(green.at<float>(i,j),g)*
                                power(blue.at<float>(i,j),b);
                    vector <double> curGrad;
                    gradvector(curIm,curGrad);
                    add_to_vector_poly(polyGrad,curGrad);
                }
            }

    red.release();
    green.release();
    blue.release();
    curIm.release();
}

void Decolor::wei_update_matrix(vector < vector <double> > &poly, vector <double> &Cg, Mat &X)
{
    Mat P = Mat(poly.size(),poly[0].size(), CV_32FC1);
    Mat A = Mat(poly.size(),poly.size(), CV_32FC1);

    for(unsigned int i =0;i<poly.size();i++)
        for(unsigned int j=0;j<poly[0].size();j++)
            P.at<float>(i,j) = (float) poly[i][j];

    Mat P_trans = P.t();
    Mat B = Mat(poly.size(),poly[0].size(), CV_32FC1);
    for(unsigned int i =0;i < poly.size();i++)
    {
        for(unsigned int j=0;j<Cg.size();j++)
            B.at<float>(i,j) = (float) (poly[i][j] * Cg[j]);
    }

    A = P*P_trans;
    solve(A, B, X, DECOMP_NORMAL);

    P.release();
    A.release();
    B.release();

}

void Decolor::wei_inti(vector < vector <int> > &comb, vector <double> &wei)
{
    vector <double> initRGB;

    initRGB.push_back( .33 );
    initRGB.push_back( .33 );
    initRGB.push_back( .33 );
    wei = product(comb,initRGB);

    vector <int> sum;

    for(unsigned int i=0;i<comb.size();i++)
        sum.push_back(comb[i][0] + comb[i][1] + comb[i][2]);

    for(unsigned int i=0;i<sum.size();i++)
    {
        if(sum[i] == 1)
            wei[i] = wei[i] * double(1);
        else
            wei[i] = wei[i] * double(0);
    }

    initRGB.clear();
    sum.clear();

}

void Decolor::grayImContruct(vector <double> &wei, Mat img, Mat &Gray)
{
    int h=img.size().height;
    int w=img.size().width;

    Mat red = Mat(img.size(),CV_32FC1);
    Mat green = Mat(img.size(),CV_32FC1);
    Mat blue = Mat(img.size(),CV_32FC1);

    for(int i=0;i<img.size().height;i++)
        for(int j=0;j<img.size().width;j++)
        {
            red.at<float>(i,j) = img.at<float>(i,j*3+2);
            green.at<float>(i,j) = img.at<float>(i,j*3+1);
            blue.at<float>(i,j) = img.at<float>(i,j*3+0);
        }

    int kk =0;

    for(int r =0;r<=order;r++)
        for(int g=0;g<=order;g++)
            for(int b=0;b<=order;b++)
                if((r + g + b) <=order && (r+g+b) > 0)
                {
                    for(int i = 0;i<h;i++)
                        for(int j=0;j<w;j++)
                            Gray.at<float>(i,j)=Gray.at<float>(i,j) +
                                (float) wei[kk]*power(red.at<float>(i,j),r)*power(green.at<float>(i,j),g)*
                                power(blue.at<float>(i,j),b);

                    kk=kk+1;
                }

    float minval = INT_MAX;
    float maxval = INT_MIN;

    for(int i=0;i<h;i++)
        for(int j =0;j<w;j++)
        {
            if(Gray.at<float>(i,j) < minval)
                minval = Gray.at<float>(i,j);

            if(Gray.at<float>(i,j) > maxval)
                maxval = Gray.at<float>(i,j);
        }

    for(int i=0;i<h;i++)
        for(int j=0;j<w;j++)
            Gray.at<float>(i,j) = (Gray.at<float>(i,j) - minval)/(maxval - minval);

    red.release();
    green.release();
    blue.release();
}
