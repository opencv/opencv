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
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

class Decolor
{
    private:
        Mat kernelx;
        Mat kernely;
        int order;

    public:
        float sigma;

        Decolor();
        static vector<double> product(const vector <Vec3i> &comb, const double initRGB[3]);
        double energyCalcu(const vector <double> &Cg, const vector < vector <double> > &polyGrad, const vector <double> &wei) const;
        void singleChannelGradx(const Mat &img, Mat& dest) const;
        void singleChannelGrady(const Mat &img, Mat& dest) const;
        void gradvector(const Mat &img, vector <double> &grad) const;
        void colorGrad(const Mat &img, vector <double> &Cg) const;
        static void add_vector(vector <Vec3i> &comb, int &idx, int r,int g,int b);
        static void add_to_vector_poly(vector < vector <double> > &polyGrad, const vector <double> &curGrad, int &idx1);
        void weak_order(const Mat &img, vector <double> &alf) const;
        void grad_system(const Mat &img, vector < vector < double > > &polyGrad,
                vector < double > &Cg, vector <Vec3i>& comb) const;
        static void wei_update_matrix(const vector < vector <double> > &poly, const vector <double> &Cg, Mat &X);
        static void wei_inti(const vector <Vec3i> &comb, vector <double> &wei);
        void grayImContruct(vector <double> &wei, const Mat &img, Mat &Gray) const;
};

double Decolor::energyCalcu(const vector <double> &Cg, const vector < vector <double> > &polyGrad, const vector <double> &wei) const
{
    const size_t size = polyGrad[0].size();
    vector <double> energy(size);
    vector <double> temp(size);
    vector <double> temp1(size);

    for(size_t i=0;i< polyGrad[0].size();i++)
    {
        double val = 0.0;
        for(size_t j =0;j<polyGrad.size();j++)
            val = val + (polyGrad[j][i] * wei[j]);
        temp[i] = val - Cg[i];
        temp1[i] = val + Cg[i];
    }

    for(size_t i=0;i<polyGrad[0].size();i++)
        energy[i] = -1.0*log(exp(-1.0*pow(temp[i],2)/sigma) + exp(-1.0*pow(temp1[i],2)/sigma));

    double sum = 0.0;
    for(size_t i=0;i<polyGrad[0].size();i++)
        sum +=energy[i];

    return (sum/polyGrad[0].size());

}

Decolor::Decolor()
{
    kernelx = Mat(1,2, CV_32FC1);
    kernely = Mat(2,1, CV_32FC1);
    kernelx.at<float>(0,0)=1.0;
    kernelx.at<float>(0,1)=-1.0;
    kernely.at<float>(0,0)=1.0;
    kernely.at<float>(1,0)=-1.0;
    order = 2;
    sigma = 0.02f;
}

vector<double> Decolor::product(const vector <Vec3i> &comb, const double initRGB[3])
{
    vector <double> res(comb.size());
    for (size_t i=0;i<comb.size();i++)
    {
        double dp = 0.0;
        for(int j=0;j<3;j++)
            dp = dp + (comb[i][j] * initRGB[j]);
        res[i] = dp;
    }
    return res;
}

void Decolor::singleChannelGradx(const Mat &img, Mat& dest) const
{
    const int w = img.size().width;
    const Point anchor(kernelx.cols - kernelx.cols/2 - 1, kernelx.rows - kernelx.rows/2 - 1);
    filter2D(img, dest, -1, kernelx, anchor, 0.0, BORDER_CONSTANT);
    dest.col(w - 1) = 0.0;
}

void Decolor::singleChannelGrady(const Mat &img, Mat& dest) const
{
    const int h = img.size().height;
    const Point anchor(kernely.cols - kernely.cols/2 - 1, kernely.rows - kernely.rows/2 - 1);
    filter2D(img, dest, -1, kernely, anchor, 0.0, BORDER_CONSTANT);
    dest.row(h - 1) = 0.0;
}

void Decolor::gradvector(const Mat &img, vector <double> &grad) const
{
    Mat dest;
    Mat dest1;
    singleChannelGradx(img,dest);
    singleChannelGrady(img,dest1);

    Mat d_trans=dest.t();
    Mat d1_trans=dest1.t();

    const int height = d_trans.size().height;
    const int width = d_trans.size().width;

    grad.resize(width * height * 2);

    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            grad[i*width + j] = d_trans.at<float>(i, j);

    const int offset = width * height;
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            grad[offset + i * width + j] = d1_trans.at<float>(i, j);
}

void Decolor::colorGrad(const Mat &img, vector <double> &Cg) const
{
    Mat lab;

    cvtColor(img,lab,COLOR_BGR2Lab);

    vector <Mat> lab_channel;
    split(lab,lab_channel);

    vector <double> ImL;
    vector <double> Ima;
    vector <double> Imb;

    gradvector(lab_channel[0],ImL);
    gradvector(lab_channel[1],Ima);
    gradvector(lab_channel[2],Imb);

    Cg.resize(ImL.size());
    for(size_t i=0;i<ImL.size();i++)
    {
        const double res = sqrt(pow(ImL[i],2) + pow(Ima[i],2) + pow(Imb[i],2))/100;
        Cg[i] = res;
    }
}

void Decolor::add_vector(vector <Vec3i> &comb, int &idx, int r,int g,int b)
{
    comb.push_back(Vec3i(r, g, b));
    idx++;
}

void Decolor::add_to_vector_poly(vector < vector <double> > &polyGrad, const vector <double> &curGrad, int &idx1)
{
    polyGrad.push_back(curGrad);
    idx1++;
}

void Decolor::weak_order(const Mat &im, vector <double> &alf) const
{
    Mat img;
    const int h = im.size().height;
    const int w = im.size().width;
    if((h + w) > 800)
    {
        const double sizefactor = double(800)/(h+w);
        resize(im, img, Size(cvRound(w*sizefactor), cvRound(h*sizefactor)));
    }
    else
    {
        img = im;
    }

    Mat curIm = Mat(img.size(),CV_32FC1);
    vector <Mat> rgb_channel;
    split(img,rgb_channel);

    vector <double> Rg, Gg, Bg;
    gradvector(rgb_channel[2],Rg);
    gradvector(rgb_channel[1],Gg);
    gradvector(rgb_channel[0],Bg);

    vector <double> t1(Rg.size()), t2(Rg.size()), t3(Rg.size());
    vector <double> tmp1(Rg.size()), tmp2(Rg.size()), tmp3(Rg.size());

    const double level = .05;

    for(size_t i=0;i<Rg.size();i++)
    {
        t1[i] = (Rg[i] > level) ? 1.0 : 0.0;
        t2[i] = (Gg[i] > level) ? 1.0 : 0.0;
        t3[i] = (Bg[i] > level) ? 1.0 : 0.0;
        tmp1[i] = (Rg[i] < -1.0*level) ? 1.0 : 0.0;
        tmp2[i] = (Gg[i] < -1.0*level) ? 1.0 : 0.0;
        tmp3[i] = (Bg[i] < -1.0*level) ? 1.0 : 0.0;
    }

    alf.resize(Rg.size());
    for(size_t i =0 ;i < Rg.size();i++)
        alf[i] = (t1[i] * t2[i] * t3[i]);

    for(size_t i =0 ;i < Rg.size();i++)
        alf[i] -= tmp1[i] * tmp2[i] * tmp3[i];
}

void Decolor::grad_system(const Mat &im, vector < vector < double > > &polyGrad,
        vector < double > &Cg, vector <Vec3i>& comb) const
{
    Mat img;
    int h = im.size().height;
    int w = im.size().width;
    if((h + w) > 800)
    {
        const double sizefactor = double(800)/(h+w);
        resize(im, img, Size(cvRound(w*sizefactor), cvRound(h*sizefactor)));
    }
    else
    {
        img = im;
    }

    h = img.size().height;
    w = img.size().width;
    colorGrad(img,Cg);

    Mat curIm = Mat(img.size(),CV_32FC1);
    vector <Mat> rgb_channel;
    split(img,rgb_channel);

    int idx = 0, idx1 = 0;
    for(int r=0 ;r <=order; r++)
        for(int g=0; g<=order;g++)
            for(int b =0; b <=order;b++)
            {
                if((r+g+b)<=order && (r+g+b) > 0)
                {
                    add_vector(comb,idx,r,g,b);
                    for(int i = 0;i<h;i++)
                        for(int j=0;j<w;j++)
                            curIm.at<float>(i,j)=static_cast<float>(
                                pow(rgb_channel[2].at<float>(i,j),r)*pow(rgb_channel[1].at<float>(i,j),g)*
                                pow(rgb_channel[0].at<float>(i,j),b));
                    vector <double> curGrad;
                    gradvector(curIm,curGrad);
                    add_to_vector_poly(polyGrad,curGrad,idx1);
                }
            }
}

void Decolor::wei_update_matrix(const vector < vector <double> > &poly, const vector <double> &Cg, Mat &X)
{
    const int size = static_cast<int>(poly.size());
    const int size0 = static_cast<int>(poly[0].size());
    Mat P = Mat(size, size0, CV_32FC1);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size0;j++)
            P.at<float>(i,j) = static_cast<float>(poly[i][j]);

    const Mat P_trans = P.t();
    Mat B = Mat(size, size0, CV_32FC1);
    for(int i =0;i < size;i++)
    {
        for(int j = 0, end = int(Cg.size()); j < end;j++)
            B.at<float>(i,j) = static_cast<float>(poly[i][j] * Cg[j]);
    }

    Mat A = P*P_trans;
    solve(A, B, X, DECOMP_NORMAL);

}

void Decolor::wei_inti(const vector <Vec3i> &comb, vector <double> &wei)
{
    double initRGB[3] = { .33, .33, .33 };

    wei = product(comb,initRGB);

    vector <int> sum(comb.size());

    for(size_t i=0;i<comb.size();i++)
        sum[i] = (comb[i][0] + comb[i][1] + comb[i][2]);

    for(size_t i=0;i<sum.size();i++)
    {
        if(sum[i] == 1)
            wei[i] = wei[i] * double(1);
        else
            wei[i] = wei[i] * double(0);
    }

    sum.clear();

}

void Decolor::grayImContruct(vector <double> &wei, const Mat &img, Mat &Gray) const
{
    const int h = img.size().height;
    const int w = img.size().width;

    vector <Mat> rgb_channel;
    split(img,rgb_channel);

    int kk =0;

    for(int r =0;r<=order;r++)
        for(int g=0;g<=order;g++)
            for(int b=0;b<=order;b++)
                if((r + g + b) <=order && (r+g+b) > 0)
                {
                    for(int i = 0;i<h;i++)
                        for(int j=0;j<w;j++)
                            Gray.at<float>(i,j)=Gray.at<float>(i,j) +
                                static_cast<float>(wei[kk])*pow(rgb_channel[2].at<float>(i,j),r)*pow(rgb_channel[1].at<float>(i,j),g)*
                                pow(rgb_channel[0].at<float>(i,j),b);

                    kk=kk+1;
                }

    double minval, maxval;
    minMaxLoc(Gray, &minval, &maxval);

    Gray -= minval;
    Gray /= maxval - minval;
}
