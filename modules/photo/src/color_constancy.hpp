#include <iostream>
#include "math.h"
#include <vector>
#include "opencv2/photo.hpp"
#include <limits>

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
        void compute_spvar(Mat &im, int sigma, Mat &Rw, Mat &Gw, Mat &Bw, Mat &sp_var);
        void general_cc(Mat &I, int njet, int mink_norm, int sigma, double &white_R, double &white_G, double &white_B, Mat &output);
        void weightedGE(Mat &input_im, int kappa, int mink_norm, int sigma, double &white_R, double &white_G, double &white_B, Mat &output);
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
    int filtersize = (int)floor(break_off_sigma*sigma + 0.5);

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
        x.ptr<float>(0)[i] = (float)filter;
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

void Constancy::compute_spvar(Mat &im, int sigma, Mat &Rw, Mat &Gw, Mat &Bw, Mat &sp_var)
{
    vector <Mat> channels;
    split(im,channels);

    Mat Rx = Mat(im.size(),CV_32FC1);
    Mat Ry = Mat(im.size(),CV_32FC1);
    Mat Gx = Mat(im.size(),CV_32FC1);
    Mat Gy = Mat(im.size(),CV_32FC1);
    Mat Bx = Mat(im.size(),CV_32FC1);
    Mat By = Mat(im.size(),CV_32FC1);
    Mat temp_x,temp_y;

    gDer(channels[2],sigma,1,0,Rx);
    gDer(channels[2],sigma,0,1,Ry);
    pow(Rx,2,temp_x);
    pow(Ry,2,temp_y);
    sqrt(temp_x + temp_y,Rw);

    gDer(channels[1],sigma,1,0,Gx);
    gDer(channels[1],sigma,0,1,Gy);
    pow(Gx,2,temp_x);
    pow(Gy,2,temp_y);
    sqrt(temp_x + temp_y,Gw);

    gDer(channels[0],sigma,1,0,Bx);
    gDer(channels[0],sigma,0,1,By);
    pow(Bx,2,temp_x);
    pow(By,2,temp_y);
    sqrt(temp_x + temp_y,Bw);

    Mat o3_x = (Rx+Gx+Bx)/sqrt(3);
    Mat o3_y = (Ry+Gy+By)/sqrt(3);

    pow(o3_x,2,temp_x);
    pow(o3_y,2,temp_y);
    sqrt(temp_x + temp_y,sp_var);
}

void Constancy::weightedGE(Mat &input_im, int kappa, int mink_norm, int sigma, double &white_R, double &white_G, double &white_B, Mat &output)
{
    int rows = input_im.rows;
    int cols = input_im.cols;
    int iter = 10;
    Mat mask_cal = Mat::zeros(rows,cols,CV_32FC1);
    double eps = std::numeric_limits<double>::epsilon();
    Mat tmp_ill = Mat::ones(1,3,CV_32FC1)*(1/sqrt(3));

    Mat final_ill = tmp_ill.clone();

    Mat tmp_image = input_im.clone();

    int flag = 1;
    vector <Mat> tmp_channels;

    while(iter && flag)
    {

        iter = iter - 1;
        split(tmp_image,tmp_channels);

        tmp_channels[2] = tmp_channels[2]/(sqrt(3)*tmp_ill.ptr<float>(0)[0]);
        tmp_channels[1] = tmp_channels[1]/(sqrt(3)*tmp_ill.ptr<float>(0)[1]);
        tmp_channels[0] = tmp_channels[0]/(sqrt(3)*tmp_ill.ptr<float>(0)[2]);

        merge(tmp_channels,tmp_image);

        Mat Rw = Mat::zeros(tmp_image.size(),CV_32FC1);
        Mat Gw = Mat::zeros(tmp_image.size(),CV_32FC1);
        Mat Bw = Mat::zeros(tmp_image.size(),CV_32FC1);
        Mat sp_var = Mat::zeros(tmp_image.size(),CV_32FC1);

        compute_spvar(tmp_image,sigma,Rw,Gw,Bw,sp_var);

        Mat out = max(Rw,max(Gw,Bw));
        Mat mask_zeros = Mat::zeros(rows,cols,CV_32FC1);
        Mat temp_zero_mask  = (out < eps)/255;
        temp_zero_mask.convertTo(mask_zeros,CV_32FC1,1);

        Mat mask_pixels = Mat::zeros(rows,cols,CV_32FC3);
        Mat max_output = max(tmp_channels[0],max(tmp_channels[1],tmp_channels[2]));
        Mat temp_out = (max_output == 255)/255;
        temp_out.convertTo(max_output,CV_32FC1,1);

        dilation33(max_output,mask_pixels);

        Mat temp = Mat(rows,cols,CV_32FC1);
        Mat mask = Mat::zeros(rows,cols,CV_32FC1);

        Mat temp_or = Mat::zeros(rows,cols,CV_32FC1);
        bitwise_or(mask_cal,mask_pixels,temp_or);
        bitwise_or(temp_or,mask_zeros,temp_or);

        Mat temp_or_output = (temp_or == 0)/255;
        temp_or_output.convertTo(temp,CV_32FC1,1);

        set_border(temp,sigma+1,0,mask);

        Mat grad_im = Mat(rows,cols,CV_32FC1);
        sqrt(Rw.mul(Rw) + Gw.mul(Gw) + Bw.mul(Bw),grad_im);

        Mat weight_map;
        pow(sp_var.mul(1/grad_im),kappa,weight_map);

        threshold(weight_map, weight_map, 1, 1, 2);

        Mat data_Rx = Mat(rows,cols,CV_32FC1);
        Mat data_Gx = Mat(rows,cols,CV_32FC1);
        Mat data_Bx = Mat(rows,cols,CV_32FC1);

        pow(Rw.mul(weight_map),mink_norm,data_Rx);
        pow(Gw.mul(weight_map),mink_norm,data_Gx);
        pow(Bw.mul(weight_map),mink_norm,data_Bx);

        Mat mask_tmp = Mat(1,rows*cols,CV_32FC1);
        Mat dataR_tmp = Mat(1,rows*cols,CV_32FC1);
        Mat dataG_tmp = Mat(1,rows*cols,CV_32FC1);
        Mat dataB_tmp = Mat(1,rows*cols,CV_32FC1);

        float sumR = 0.0, sumG = 0.0,sumB = 0.0;
        int k =0;
        for(int j = 0;j<cols;j++)
        {
            for(int i=0;i<rows;i++)
            {
                dataR_tmp.ptr<float>(0)[k] = data_Rx.ptr<float>(i)[j];
                dataG_tmp.ptr<float>(0)[k] = data_Gx.ptr<float>(i)[j];
                dataB_tmp.ptr<float>(0)[k] = data_Bx.ptr<float>(i)[j];
                mask_tmp.ptr<float>(0)[k] = mask.ptr<float>(i)[j];
                k = k+1;
            }
        }

        for(k = 0;k<rows*cols;k++)
        {
            if(mask_tmp.ptr<float>(0)[k] == 1)
            {
                sumR = sumR + dataR_tmp.ptr<float>(0)[k];
                sumG = sumG + dataG_tmp.ptr<float>(0)[k];
                sumB = sumB + dataB_tmp.ptr<float>(0)[k];
            }
        }

        float t1 = (float)1/mink_norm;

        tmp_ill.ptr<float>(0)[0] = pow(sumR,t1);
        tmp_ill.ptr<float>(0)[1] = pow(sumG,t1);
        tmp_ill.ptr<float>(0)[2] = pow(sumB,t1);

        Mat norm1;
        pow(tmp_ill,2,norm1);
        double sum_norm1 = sum(norm1)[0];
        tmp_ill = tmp_ill/sqrt(sum_norm1);
        final_ill = final_ill.mul(tmp_ill);
        Mat norm;
        pow(final_ill,2,norm);
        double sum_norm = sum(norm)[0];

        final_ill = final_ill/sqrt(sum_norm);

        Mat break_cond = (tmp_ill*( 1/sqrt(3)*(Mat::ones(3,1,CV_32FC1))));
        if ( ( acos(break_cond.ptr<float>(0)[0])/CV_PI*180 ) < 0.05 )
        {
            flag = 0;
        }
    }

    white_R = (double)final_ill.ptr<float>(0)[0];
    white_G = (double)final_ill.ptr<float>(0)[1];
    white_B = (double)final_ill.ptr<float>(0)[2];

    vector <Mat> out_planes;
    split(output,out_planes);

    vector <Mat> rgb_planes;
    split(input_im,rgb_planes);

    out_planes[2] = rgb_planes[2]/(sqrt(3)*(final_ill.ptr<float>(0)[0]));
    out_planes[1] = rgb_planes[1]/(sqrt(3)*(final_ill.ptr<float>(0)[1]));
    out_planes[0] = rgb_planes[0]/(sqrt(3)*(final_ill.ptr<float>(0)[2]));

    merge(out_planes,output);
    output.convertTo(output,CV_8UC3,1);
}

void Constancy::general_cc(Mat &I, int njet, int mink_norm, int sigma, double &white_R, double &white_G, double &white_B, Mat &output)
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
