#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core_c.h>

using namespace std;
using namespace cv;

class Intrinsic
{
    public:
        void decompose(Mat &rgbIm, Mat &ref, Mat &shade, int wd, int iterNum, float rho);
};

void Intrinsic::decompose(Mat &rgbIm, Mat &ref, Mat &shade, int wd, int iterNum, float rho)
{
    int n = rgbIm.rows;
    int m = rgbIm.cols;
    int channels = rgbIm.channels();
    int imgsize = n*m;
    float epsilon = 1/10000;

    Mat yplane = Mat(n,m,CV_32FC3);
    cvtColor(rgbIm,yplane,COLOR_BGR2YCrCb);
    rgbIm = rgbIm + epsilon;

    Mat temp;
    pow(rgbIm,2,temp);
    vector <Mat> planes;
    split(temp,planes);
    Mat rgbIm2 = planes[0] + planes[1] + planes[2];

    pow(rgbIm2,0.5,temp);
    Mat out;
    Mat in[] = {temp, temp, temp};
    merge(in, 3, out);

    Mat rgbIm_c;
    divide(rgbIm,out,rgbIm_c);
    int column = std::pow(2*wd+1,2);

    // luminlances in a local window
    Mat gvals = Mat::zeros(1,column,CV_32FC1);
    // angles in a local window
    Mat t_cvals = Mat::zeros(column,3,CV_32FC1);

    int num_w = column -1;
    // Weight Matrix
    Mat weight = Mat::zeros(imgsize,num_w,CV_32FC1);
    Mat n_idx = Mat::ones(imgsize,num_w,CV_32FC1);
    Mat pixelIdx = Mat(n,m,CV_32FC1);
    for(int j = 0;j < m;j++)
        for(int i = 0, idx = j*n;i<n;i++,idx++)
            pixelIdx.at<float>(i,j) = (float)idx;


    int len = 0;
    int tlen = 0;
    for(int j = 0;j<m;j++)
    {
        for(int i = 0;i<n;i++)
        {
            tlen = 0;
            for(int ii = max(0,i-wd);ii<min(i+wd+1,n);ii++)
                for(int jj = max(0,j-wd);jj<min(j+wd+1,m);jj++)
                {

                    if(ii!=i || jj!=j)
                    {
                        gvals.at<float>(0,tlen) = yplane.at<float>(ii,jj*channels+0);
                        for(int c=0; c <channels; c++)
                            t_cvals.at<float>(tlen,c) = rgbIm_c.at<float>(ii,jj*channels+c);
                        n_idx.at<float>(len,tlen) = pixelIdx.at<float>(ii,jj);
                        tlen = tlen +1;
                    }
                }

            len= len +1;

            float t_val = yplane.at<float>(i,j*channels+0);
            gvals.at<float>(0,tlen) = t_val;
            Scalar mean_gvals = mean(gvals(Range::all(),Range(0,tlen+1)));
            Mat sub_mean = gvals(Range::all(),Range(0,tlen+1)) - mean_gvals;
            pow(sub_mean,2,sub_mean);
            Scalar c_var = mean(sub_mean);
            double csig = c_var[0] * 0.6;

            Mat subtract = gvals(Range::all(),Range(0,tlen)) - t_val;
            pow(subtract,2,subtract);
            double minVal;
            cv::minMaxIdx(subtract, &minVal,NULL);
            double mgv = minVal;
            if(csig < (-1*mgv/log(.01)))
                csig = -1*mgv/log(.01);
            if(csig < .000002)
                csig = .000002;

            Mat t_cval = Mat(1,3,CV_32FC1);
            for(int c = 0;c<channels;c++)
                t_cval.at<float>(0,c) = rgbIm_c.at<float>(i,j*channels+c);

            Mat rep_mat;
            repeat(t_cval,tlen,1,rep_mat);
            multiply(t_cvals(Range(0,tlen),Range::all()),rep_mat,rep_mat);

            Mat cvals;
            reduce(rep_mat,cvals,1,CV_REDUCE_SUM);

            for(int y = 0;y<cvals.rows;y++)
            {
                if(cvals.at<float>(0,y) > 1)
                {
                    cvals.at<float>(0,y) = 0;
                }
                else
                {
                    cvals.at<float>(0,y) = acos(cvals.at<float>(0,y));
                }
            }

            Mat power_cvals_diff;
            pow(cvals - mean(cvals),2,power_cvals_diff);
            Scalar c_var_cvals = mean(power_cvals_diff);

            double csig_cvals = c_var_cvals[0] * 0.6;
            Mat power_cvals;
            pow(cvals-1,2,power_cvals);
            double minVal1;
            cv::minMaxIdx(power_cvals, &minVal1,NULL);
            double mgv_cvals = minVal1;
            if(csig_cvals < (-1*mgv_cvals/log(.01)))
                csig_cvals = -1*mgv_cvals/log(.01);
            if(csig_cvals < .000002)
                csig_cvals = .000002;


            Mat cvals_t = cvals.t();
            pow(cvals_t,2,cvals_t);
            divide(cvals_t,pow(csig_cvals,2),cvals_t);

            Mat power_gvals_diff;
            pow(gvals(Range::all(),Range(0,tlen)) - t_val,2,power_gvals_diff);
            divide(power_gvals_diff,csig,power_gvals_diff);
            exp(-1*(power_gvals_diff + cvals_t),gvals(Range::all(),Range(0,tlen)));

            Scalar tmp_sum = sum(gvals(Range::all(),Range(0,tlen)));
            divide(gvals(Range::all(),Range(0,tlen)),tmp_sum,gvals(Range::all(),Range(0,tlen)));
            for(int y = 0; y < tlen;y++)
                weight.at<float>(len-1,y) = gvals.at<float>(0,y);

        }
    }
}
