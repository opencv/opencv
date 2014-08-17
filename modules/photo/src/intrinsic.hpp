#include <iostream>
#include "math.h"
#include <vector>
#include <opencv2/core/core_c.h>

using namespace std;
using namespace cv;

class Intrinsic
{
    private:
        int n,m,channels,imgsize,column,num_w;
        float epsilon;
        Mat n_idx;
    public:
        void init(Mat &rgbIm);
        double calc(double val, double temp_val);
        void compute(Mat &rgbIm, Mat &rgbIm2, Mat &weight, Mat &ref, Mat &shade, int iterNum, float rho);
        void decompose(Mat &rgbIm, Mat &ref, Mat &shade, int wd, int iterNum, float rho);
};

void Intrinsic::init(Mat &rgbIm)
{
    n = rgbIm.rows;
    m = rgbIm.cols;
    channels = rgbIm.channels();
    imgsize = n*m;
    epsilon = 1/10000;
}

inline double Intrinsic::calc(double val, double temp_val)
{
    if(val < temp_val)
        val = temp_val;
    if(val < .000002)
        val = .000002;

    return val;
}

void Intrinsic::compute(Mat &rgbIm, Mat &rgbIm2, Mat &weight, Mat &ref, Mat &shade, int iterNum, float rho)
{
    Mat R = Mat::ones(imgsize,3,CV_32FC1)*0.5;
    Mat inv_S_r = Mat::ones(imgsize,1,CV_32FC1)*2.0;
    Mat rgbIm_reshape = Mat(imgsize,3,CV_32FC1);
    int length = 0;
    for(int j = 0;j<m;j++)
    {
        for(int i = 0;i<n;i++)
        {
            Vec3f *rgb_temp = rgbIm.ptr<Vec3f>(i);
            for(int c=0;c<channels;c++)
                rgbIm_reshape.ptr<float>(length)[c] = rgb_temp[j][c];
            length = length +1;
        }
    }

    length = 0;
    Mat rgbIm2_reshape = Mat(imgsize,1,CV_32FC1);
    for(int j = 0;j<m;j++)
        for(int i = 0;i<n;i++)
        {
            rgbIm2_reshape.ptr<float>(length)[0] = rgbIm2.ptr<float>(i)[j];
            length = length +1;
        }

    for(int k = 0;k<iterNum;k++)
    {
        Mat inv_S;
        repeat(inv_S_r,1,3,inv_S);
        Mat si;
        multiply(inv_S,rgbIm_reshape,si);
        int len = 0;

        for(int j = 0;j<m;j++)
        {
            for(int i = 0;i<n;i++)
            {
                len = len+1;
                Mat sumR;
                Mat temp = Mat(num_w,3,CV_32FC1);
                for(int l =0;l<num_w;l++)
                {
                    for(int y =0;y<3;y++)
                    {
                        temp.ptr<float>(l)[y] = R.ptr<float>(n_idx.at<float>(len-1,l))[y];
                    }
                }

                sumR = weight(Range(len-1,len),Range::all())*temp + si(Range(len-1,len),Range::all());
                R(Range(len-1,len), Range::all()) = (1-rho)*R(Range(len-1,len), Range::all()) + 0.5*rho*sumR;
                for(int y =0;y<R.cols;y++)
                {
                    if(R.ptr<float>(len-1)[y] > 1)
                        R.ptr<float>(len-1)[y] = 1.0;
                }
                for(int y =0;y<R.cols;y++)
                {
                    if(R.ptr<float>(len-1)[y] < epsilon)
                        R.ptr<float>(len-1)[y] = epsilon;
                }
            }
        }

        Mat temp1;
        multiply(rgbIm_reshape,R,temp1);
        reduce(temp1,temp1,1,CV_REDUCE_SUM);
        divide(temp1,rgbIm2_reshape,temp1);
        inv_S_r = (1-rho)*inv_S_r + rho*temp1;
        for(int y = 0;y<inv_S_r.rows;y++)
        {
            if(inv_S_r.ptr<float>(y)[0] < 1)
                inv_S_r.ptr<float>(y)[0] = 1;
        }

    }

    Mat ref32f = Mat(n,m,CV_32FC3);
    length = 0;
    for(int j = 0;j<m;j++)
    {
        for(int i = 0;i<n;i++)
        {
            Vec3f *ref32f_ptr = ref32f.ptr<Vec3f>(i);

            for(int c = 0;c<channels;c++)
                ref32f_ptr[j][c] = R.ptr<float>(length)[c];
            length = length +1;
        }
    }

    cvtColor(ref32f,ref32f,COLOR_RGB2BGR);
    ref32f.convertTo(ref,CV_8UC3,255);
    divide(1,inv_S_r,inv_S_r);
    Mat shade32f = Mat(n,m,CV_32FC1);
    length = 0;
    for(int j = 0;j<m;j++)
    {
        for(int i = 0;i<n;i++)
        {
            shade32f.ptr<float>(i)[j] = inv_S_r.ptr<float>(length)[0];
            length = length +1;
        }
    }
    shade32f.convertTo(shade,CV_8UC1,255);
}

void Intrinsic::decompose(Mat &rgbIm, Mat &ref, Mat &shade, int wd, int iterNum, float rho)
{
    init(rgbIm);
    Mat yplane = Mat(n,m,CV_32FC3);
    cvtColor(rgbIm,yplane,COLOR_BGR2YUV);
    cvtColor(rgbIm,rgbIm,COLOR_BGR2RGB);
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
    column = std::pow(2*wd+1,2);

    // luminlances in a local window
    Mat gvals = Mat::zeros(1,column,CV_32FC1);
    // angles in a local window
    Mat t_cvals = Mat::zeros(column,3,CV_32FC1);

    num_w = column -1;
    // weight Matrix
    Mat weight = Mat::zeros(imgsize,num_w,CV_32FC1);
    n_idx = Mat::ones(imgsize,num_w,CV_32FC1);
    Mat pixelIdx = Mat(n,m,CV_32FC1);
    for(int j = 0;j < m;j++)
    {
        for(int i = 0, idx = j*n;i<n;i++,idx++)
        {
            pixelIdx.ptr<float>(i)[j] = (float)idx;
        }
    }

    int len = 0;
    int tlen = 0;
    double log_val = log(.01);
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
                        Vec3f *yplane_ptr = yplane.ptr<Vec3f>(ii);
                        Vec3f *rgbIm_c_ptr = rgbIm_c.ptr<Vec3f>(ii);
                        gvals.ptr<float>(0)[tlen] = yplane_ptr[jj][0];
                        for(int c=0; c <channels; c++)
                            t_cvals.ptr<float>(tlen)[c] = rgbIm_c_ptr[jj][c];
                        n_idx.ptr<float>(len)[tlen] = pixelIdx.ptr<float>(ii)[jj];
                        tlen = tlen +1;
                    }
                }

            len= len +1;

            float t_val = yplane.ptr<Vec3f>(i)[j][0];
            gvals.ptr<float>(0)[tlen] = t_val;
            Scalar mean_gvals = mean(gvals(Range::all(),Range(0,tlen+1)));
            Mat sub_mean = gvals(Range::all(),Range(0,tlen+1)) - mean_gvals;
            pow(sub_mean,2,sub_mean);
            Scalar c_var = mean(sub_mean);
            double csig = c_var[0] * 0.6;

            Mat subtract = gvals(Range::all(),Range(0,tlen)) - t_val;
            pow(subtract,2,subtract);

            double minVal;
            cv::minMaxIdx(subtract, &minVal,NULL);

            double temp_val = -1*minVal/log_val;
            csig = calc(csig,temp_val);

            Mat t_cval = Mat(1,3,CV_32FC1);
            for(int c = 0;c<channels;c++)
                t_cval.ptr<float>(0)[c] = rgbIm_c.ptr<Vec3f>(i)[j][c];

            Mat rep_mat;
            repeat(t_cval,tlen,1,rep_mat);
            multiply(t_cvals(Range(0,tlen),Range::all()),rep_mat,rep_mat);

            Mat cvals;
            reduce(rep_mat,cvals,1,CV_REDUCE_SUM);

            for(int y = 0;y<cvals.rows;y++)
            {
                if(cvals.ptr<float>(0)[y] > 1)
                {
                    cvals.ptr<float>(0)[y] = 0;
                }
                else
                {
                    cvals.ptr<float>(0)[y] = acos(cvals.ptr<float>(0)[y]);
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

            double temp_val1 = -1*minVal1/log_val;
            csig_cvals = calc(csig_cvals,temp_val1);

            Mat cvals_t = cvals.t();
            pow(cvals_t,2,cvals_t);
            divide(cvals_t,pow(csig_cvals,2),cvals_t);

            Mat power_gvals_diff;
            pow(gvals(Range::all(),Range(0,tlen)) - t_val,2,power_gvals_diff);
            divide(power_gvals_diff,csig,power_gvals_diff);
            exp(-1*(power_gvals_diff + cvals_t),gvals(Range::all(),Range(0,tlen)));

            Scalar tmp_sum = sum(gvals(Range::all(),Range(0,tlen)));
            divide(gvals(Range::all(),Range(0,tlen)),tmp_sum,gvals(Range::all(),Range(0,tlen)));
            weight(Range(len-1,len),Range(0,tlen)) = gvals(Range::all(),Range(0,tlen))*1;

        }
    }

    compute(rgbIm,rgbIm2,weight,ref,shade,iterNum,rho);
}
