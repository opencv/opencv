#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class Constancy
{
    public:
        void set_border(Mat &in,int width, int method, Mat &out);
        void dilation33(Mat &in, Mat &out);
        void fill_border(Mat &in, int bw, Mat &out);
        void gDer(Mat &f, int sigma, int iorder, int jorder, Mat &H);
        void normDerivative(Mat &in, int sigma, int order, Mat &Rw, Mat &Gw, Mat &Bw);
        void general_cc(Mat &I, int njet, int mink_norm, int sigma, float &white_R, float &white_G, float &white_B, Mat &output);
};

void Constancy::set_border(Mat &in,int width, int method, Mat &out)
{
    int rows = in.rows;
    int cols = in.cols;
    Mat temp = Mat::ones(rows,cols,CV_32FC1);

    Mat y = Mat::zeros(rows,cols,CV_32FC1);
    Mat x = Mat::zeros(rows,cols,CV_32FC1);
    int border_x = cols - width;

    Mat temp1 = x.clone();
    Mat temp2 = x.clone();
    for(int j =0;j<border_x;j++)
    {
        for(int i=0;i<rows;i++)
        {
            temp1.ptr<float>(i)[j] = 1;
        }
    }

    for(int j = width; j < cols;j++)
    {
        for(int i = 0;i<rows;i++)
        {
            temp2.ptr<float>(i)[j] = 1;
        }
    }

    bitwise_and(temp1,temp2,x);

    temp1 = y.clone();
    temp2 = y.clone();

    int border_y = rows - width;

    for(int i =0;i<border_y;i++)
    {
        for(int j=0;j<cols;j++)
        {
            temp1.ptr<float>(i)[j] = 1;
        }
    }

    for(int i = width; i < rows;i++)
    {
        for(int j = 0;j<cols;j++)
        {
            temp2.ptr<float>(i)[j] = 1;
        }
    }

    bitwise_and(temp1,temp2,y);

    temp = temp.mul(x);
    temp = temp.mul(y);
    out = temp.mul(in);

    if(method == 1)
    {
        out = out + (sum(out)[0]/sum(temp)[0])*(Mat::ones(rows,cols,CV_32FC1) - temp);
    }
}

void Constancy::dilation33(Mat &in, Mat &out)
{
    int hh = in.rows;
    int ll = in.cols;

    vector <Mat> out_planes;
    split(out,out_planes);

    out_planes[2](Range(0,hh-1),Range::all()) = in(Range(2-1,hh),Range::all())*1; 
    out_planes[2](Range(hh-1,hh),Range::all()) = in(Range(hh-1,hh),Range::all())*1; 
    out_planes[1] = in;
    out_planes[0](Range(0,1),Range::all()) = in(Range(0,1),Range::all())*1; 
    out_planes[0](Range(1,hh),Range::all()) = in(Range(0,hh-1),Range::all())*1;
    
    Mat out2 = max(out_planes[0],max(out_planes[1],out_planes[2]));
    
    out_planes[2](Range::all(),Range(0,ll-1)) = out2(Range::all(),Range(2-1,ll))*1; 
    out_planes[2](Range::all(),Range(ll-1,ll)) = out2(Range::all(),Range(ll-1,ll))*1; 
    out_planes[1] = out2;
    out_planes[0](Range::all(),Range(0,1)) = out2(Range::all(),Range(0,1))*1; 
    out_planes[0](Range::all(),Range(1,ll)) = out2(Range::all(),Range(0,ll-1))*1; 

    out = max(out_planes[0],max(out_planes[1],out_planes[2]));

}

void Constancy::fill_border(Mat &in, int bw, Mat &out)
{
    int hh = in.rows;
    int ww = in.cols;
    int dd = in.channels();

    if(dd == 1)
    {
        Mat temp = Mat::ones(bw,bw,CV_32FC1);

        out(Range(0,bw),Range(0,bw)) = (Mat::ones(bw,bw,CV_32FC1))*in.at<float>(0,0);
        out(Range(bw+hh+1-1,2*bw+hh),Range(0,bw)) = Mat::ones(bw,bw,CV_32FC1)*in.at<float>(hh-1,0);
        out(Range(0,bw),Range(bw+1+ww-1,2*bw+ww)) = Mat::ones(bw,bw,CV_32FC1)*in.at<float>(0,ww-1);
        out(Range(bw+hh+1-1,2*bw+hh),Range(bw+1+ww-1,2*bw+ww)) = Mat::ones(bw,bw,CV_32FC1)*in.at<float>(hh-1,ww-1);
        out(Range(bw+1-1,bw+hh),Range(bw+1-1,bw+ww)) = in*1;
        out(Range(0,bw),Range(bw+1-1,bw+ww)) = Mat::ones(bw,1,CV_32FC1)*(in(Range(0,1),Range::all()));
        out(Range(bw+hh+1-1,2*bw+hh),Range(bw+1-1,bw+ww)) = Mat::ones(bw,1,CV_32FC1)*(in(Range(hh-1,hh),Range::all()));
        out(Range(bw+1-1,bw+hh),Range(0,bw)) = (in(Range::all(),Range(0,1)))*(Mat::ones(1,bw,CV_32FC1));
        out(Range(bw+1-1,bw+hh),Range(bw+ww+1-1,2*bw+ww)) = (in(Range::all(),Range(ww-1,ww)))*(Mat::ones(1,bw,CV_32FC1));
    }
    else
    {
        vector <Mat> out_channels;
        vector <Mat> in_channels;
        split(out,out_channels);
        split(in,in_channels);

        for(int ii =0;ii<dd;ii++)
        {
            out_channels[ii](Range(0,bw),Range(0,bw)) = (Mat::ones(bw,bw,CV_32FC1))*(in_channels[ii].at<float>(0,0));
            out_channels[ii](Range(bw+hh+1-1,2*bw+hh),Range(0,bw)) = Mat::ones(bw,bw,CV_32FC1)*(in_channels[ii].at<float>(hh-1,0));
            out_channels[ii](Range(0,bw),Range(bw+1+ww-1,2*bw+ww)) = Mat::ones(bw,bw,CV_32FC1)*(in_channels[ii].at<float>(0,ww-1));
            out_channels[ii](Range(bw+hh+1-1,2*bw+hh),Range(bw+1+ww-1,2*bw+ww)) = 
                Mat::ones(bw,bw,CV_32FC1)*(in_channels[ii].at<float>(hh-1,ww-1));
            out_channels[ii](Range(bw+1-1,bw+hh),Range(bw+1-1,bw+ww)) = in_channels[ii]*1;
            out_channels[ii](Range(0,bw),Range(bw+1-1,bw+ww)) = Mat::ones(bw,1,CV_32FC1)*(in_channels[ii](Range(0,1),Range::all()));
            out_channels[ii](Range(bw+hh+1-1,2*bw+hh),Range(bw+1-1,bw+ww)) = Mat::ones(bw,1,CV_32FC1)*(in_channels[ii](Range(hh-1,hh),Range::all()));
            out_channels[ii](Range(bw+1-1,bw+hh),Range(0,bw)) = 
                (in_channels[ii](Range::all(),Range(0,1)))*(Mat::ones(1,bw,CV_32FC1));
            out_channels[ii](Range(bw+1-1,bw+hh),Range(bw+ww+1-1,2*bw+ww)) = 
                (in_channels[ii](Range::all(),Range(ww-1,ww)))*(Mat::ones(1,bw,CV_32FC1));
        }
        merge(out_channels,out);
    }
}

void Constancy::gDer(Mat &f, int sigma, int iorder, int jorder, Mat &H)
{
    float break_off_sigma = 3;
    int filtersize = floor(break_off_sigma*sigma + 0.5);

   Mat temp;
   if(f.channels() == 1)
       temp = Mat::zeros(f.rows+filtersize*2,f.cols+filtersize*2,CV_32FC1);
   else
       temp = Mat::zeros(f.rows+filtersize*2,f.cols+filtersize*2,CV_32FC3);

    fill_border(f,filtersize,temp);
    Mat x = Mat(1,2*filtersize+1,CV_32FC1);

    int filter = -1*filtersize;

    for(int i =0;i<x.cols;i++)
    {
        x.ptr<float>(0)[i] = filter;
        filter = filter+1;
    }

    Mat gauss;
    exp((x.mul(x))/(-2 * sigma * sigma) ,gauss);
    gauss = 1/(sqrt(2 * CV_PI) * sigma)*gauss; 
    Mat Gx = Mat(gauss.size(),CV_32FC1);;
    if(iorder == 0)
    {
        Gx = gauss/sum(gauss)[0];
    }
    else if(iorder == 1)
    {
        Gx = (-1*x/pow(sigma,2)).mul(gauss);
        Gx  =  Gx/(sum(x.mul(Gx))[0]);
    }
    else if(iorder == 2)
    {
        Gx = (x.mul(x)/pow(sigma,4)-1/pow(sigma,2)).mul(gauss);
        Gx = Gx-sum(Gx)[0]/(2*filtersize+1);
        Gx = Gx/sum(0.5*x.mul(x.mul(Gx)))[0];
    }

    Point anchor(Gx.cols - Gx.cols/2 - 1, Gx.rows - Gx.rows/2 - 1);
    filter2D(temp, H, CV_32F, Gx,anchor,0.0,BORDER_CONSTANT);

    Mat Gy = Mat(gauss.size(),CV_32FC1);
    if(jorder == 0)
    {
        Gy = gauss/sum(gauss)[0];
    }
    else if(jorder == 1)
    {
        Gy = -(x/pow(sigma,2)).mul(gauss);
        Gy  =  Gy/(sum(x.mul(Gy))[0]);
    }
    else if(jorder == 2)
    {
        Gy = (x.mul(x)/pow(sigma,4)-1/pow(sigma,2)).mul(gauss);
        Gy = Gy-sum(Gy)[0]/(2*filtersize+1);
        Gy = Gy/sum(0.5*x.mul(x.mul(Gy)))[0];
    }

    Mat Gy_t = Gy.t();
    Point anchor1(Gy_t.cols - Gy_t.cols/2 - 1, Gy_t.rows - Gy_t.rows/2 - 1);
    filter2D(H, H, CV_32F, Gy_t,anchor1,0.0,BORDER_CONSTANT);

    H = H(Range(filtersize+1-1,H.rows-filtersize),Range(filtersize+1-1,H.cols-filtersize))*1;

}

void Constancy::normDerivative(Mat &in, int sigma, int order, Mat &Rw, Mat &Gw, Mat &Bw)
{
    int rows = in.rows;
    int cols = in.cols;

    vector <Mat> channels;
    split(in,channels);
    Mat temp1,temp2;

    if(order == 1)
    {
        Mat Rx = Mat(rows,cols,CV_32FC1);
        Mat Ry = Mat(rows,cols,CV_32FC1);
        Mat Gx = Mat(rows,cols,CV_32FC1);
        Mat Gy = Mat(rows,cols,CV_32FC1);
        Mat Bx = Mat(rows,cols,CV_32FC1);
        Mat By = Mat(rows,cols,CV_32FC1);

        gDer(channels[2],sigma,1,0,Rx);
        gDer(channels[2],sigma,0,1,Ry);
        pow(Rx,2,temp1);
        pow(Ry,2,temp2);
        sqrt(temp1 + temp2,Rw);

        gDer(channels[1],sigma,1,0,Gx);
        gDer(channels[1],sigma,0,1,Gy);
        pow(Gx,2,temp1);
        pow(Gy,2,temp2);
        sqrt(temp1 + temp2,Gw);

        gDer(channels[0],sigma,1,0,Bx);
        gDer(channels[0],sigma,0,1,By);
        pow(Bx,2,temp1);
        pow(By,2,temp2);
        sqrt(temp1+temp2,Bw);
    }
    if(order == 2)
    {
        Mat Rxx = Mat(rows,cols,CV_32FC1);
        Mat Ryy = Mat(rows,cols,CV_32FC1);
        Mat Rxy = Mat(rows,cols,CV_32FC1);
        Mat Gxx = Mat(rows,cols,CV_32FC1);
        Mat Gyy = Mat(rows,cols,CV_32FC1);
        Mat Gxy = Mat(rows,cols,CV_32FC1);
        Mat Bxx = Mat(rows,cols,CV_32FC1);
        Mat Byy = Mat(rows,cols,CV_32FC1);
        Mat Bxy = Mat(rows,cols,CV_32FC1);

        gDer(channels[2],sigma,2,0,Rxx);
        gDer(channels[2],sigma,0,2,Ryy);
        gDer(channels[2],sigma,1,1,Rxy);
        sqrt(Rxx.mul(Rxx) + 4*Rxy.mul(Rxy) + Ryy.mul(Ryy),Rw);

        gDer(channels[1],sigma,2,0,Gxx);
        gDer(channels[1],sigma,0,2,Gyy);
        gDer(channels[1],sigma,1,1,Gxy);
        sqrt(Gxx.mul(Gxx) + 4*Gxy.mul(Gxy) + Gyy.mul(Gyy),Gw);

        gDer(channels[2],sigma,2,0,Bxx);
        gDer(channels[2],sigma,0,2,Byy);
        gDer(channels[2],sigma,1,1,Bxy);
        sqrt(Bxx.mul(Bxx) + 4*Bxy.mul(Bxy) + Byy.mul(Byy),Bw);
    }
}

void Constancy::general_cc(Mat &I, int njet, int mink_norm, int sigma, float &white_R, float &white_G, float &white_B, Mat &output)
{
    int rows = I.rows;
    int cols = I.cols;

    Mat mask_im = Mat::zeros(rows,cols,CV_32FC1);
    int saturation_threshold = 255;
    
    vector <Mat> planes;
    vector <Mat> channels;

    split(I,planes);
    split(I,channels);

    Mat mask_im2 = Mat::zeros(rows,cols,CV_32FC3);
    Mat out = max(planes[0],max(planes[1],planes[2]));
    Mat temp = Mat::zeros(rows,cols,CV_32FC1);
    for(int i = 0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(out.ptr<float>(i)[j] >=saturation_threshold)
                temp.ptr<float>(i)[j] = 1;
            else
                temp.ptr<float>(i)[j] = 0;
        }
    }
    dilation33(temp,mask_im2);
    mask_im2 = mask_im2 + mask_im;
    for(int i = 0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            if(mask_im2.ptr<float>(i)[j] == 0)
                mask_im2.ptr<float>(i)[j] = 1;
            else
                mask_im2.ptr<float>(i)[j] = 0;
        }
    }
    set_border(mask_im2,sigma+1,0,mask_im2);

    output = I.clone();
    if(njet == 0)
    {
        if(sigma!=0)
        {
            gDer(planes[2],sigma,0,0,planes[2]);
            gDer(planes[1],sigma,0,0,planes[1]);
            gDer(planes[0],sigma,0,0,planes[0]);
        }
    }

    if(njet > 0)
    {
        normDerivative(I,sigma,njet,channels[2],channels[1],channels[0]);
        merge(channels,I);
    }

    I = abs(I);

    if(mink_norm !=-1)
    {
        Mat kleur = Mat(rows,cols,CV_32FC3);
        pow(I,mink_norm,kleur);
        vector <Mat> kleur_channel;
        split(kleur,kleur_channel); 

        float t1 = (float)1/mink_norm;
        white_R = pow(sum(kleur_channel[2].mul(mask_im2))[0],t1);
        white_G = pow(sum(kleur_channel[1].mul(mask_im2))[0],t1);
        white_B = pow(sum(kleur_channel[0].mul(mask_im2))[0],t1);

        float som = sqrt(pow(white_R,2) + pow(white_G,2) + pow(white_B,2));

        white_R = white_R/som;
        white_G = white_G/som;
        white_B = white_B/som;

    }
    else
    {
        vector <Mat> RGB_channels;
        split(I,RGB_channels);

        double maxVal; 
        minMaxLoc(RGB_channels[2].mul(mask_im2), NULL, &maxVal);
        white_R = maxVal;
        minMaxLoc(RGB_channels[1].mul(mask_im2), NULL, &maxVal);
        white_G = maxVal;
        minMaxLoc(RGB_channels[0].mul(mask_im2), NULL, &maxVal);
        white_B = maxVal;

        float som = sqrt(pow(white_R,2) + pow(white_G,2) + pow(white_B,2));

        white_R = white_R/som;
        white_G = white_G/som;
        white_B = white_B/som;
    }

    vector <Mat> out_channels;
    split(output,out_channels);

    out_channels[2] = out_channels[2]/(white_R*sqrt(3));
    out_channels[1] = out_channels[1]/(white_G*sqrt(3));
    out_channels[0] = out_channels[0]/(white_B*sqrt(3));

    merge(out_channels,output);
}

