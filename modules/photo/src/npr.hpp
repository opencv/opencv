#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <limits>
#include "math.h"

using namespace std;
using namespace cv;

double myinf = std::numeric_limits<double>::infinity();

class Domain_Filter
{
	public:
		Mat ct_H, ct_V, horiz, vert, O, O_t, lower_idx, upper_idx;
		void init(const Mat &img, int flags, float sigma_s, float sigma_r);
		void getGradientx( const Mat &img, Mat &gx);
		void getGradienty( const Mat &img, Mat &gy);
		void diffx(const Mat &img, Mat &temp);
		void diffy(const Mat &img, Mat &temp);
		void compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius);
		void compute_Rfilter(Mat &O, Mat &horiz, float sigma_h);
		void compute_NCfilter(Mat &O, Mat &horiz, Mat &psketch, float radius);
		void filter(const Mat &img, Mat &res, float sigma_s, float sigma_r, int flags);
};

void Domain_Filter::diffx(const Mat &img, Mat &temp)
{
	int channel = img.channels();

	for(int i = 0; i < img.size().height; i++)
		for(int j = 0; j < img.size().width-1; j++)
		{
			for(int c =0; c < channel; c++)
			{
				temp.at<float>(i,j*channel+c) = 
					img.at<float>(i,(j+1)*channel+c) - img.at<float>(i,j*channel+c);
			}
		}
}

void Domain_Filter::diffy(const Mat &img, Mat &temp)
{
	int channel = img.channels();

	for(int i = 0; i < img.size().height-1; i++)
		for(int j = 0; j < img.size().width; j++)
		{
			for(int c =0; c < channel; c++)
			{
				temp.at<float>(i,j*channel+c) = 
					img.at<float>((i+1),j*channel+c) - img.at<float>(i,j*channel+c);
			}
		}
}

void Domain_Filter::getGradientx( const Mat &img, Mat &gx)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				gx.at<float>(i,j*channel+c) =
					img.at<float>(i,(j+1)*channel+c) - img.at<float>(i,j*channel+c);
			}
}
void Domain_Filter::getGradienty( const Mat &img, Mat &gy)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				gy.at<float>(i,j*channel+c) =
					img.at<float>(i+1,j*channel+c) - img.at<float>(i,j*channel+c);

			}
}

void Domain_Filter::compute_Rfilter(Mat &output, Mat &hz, float sigma_h)
{

	float a;

	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	a = exp(-sqrt(2) / sigma_h);

	Mat temp = Mat(h,w,CV_32FC3);

	for(int i =0; i < h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				temp.at<float>(i,j*channel+c) = output.at<float>(i,j*channel+c);


	Mat V = Mat(h,w,CV_32FC1);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			V.at<float>(i,j) = pow(a,hz.at<float>(i,j));


	for(int i=0; i<h; i++)
	{
		for(int j =1; j < w; j++)
		{
			for(int c = 0; c<channel; c++)
			{
				temp.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c) + 
					(temp.at<float>(i,(j-1)*channel+c) - temp.at<float>(i,j*channel+c)) * V.at<float>(i,j);
			}
		}
	}
				
	for(int i=0; i<h; i++)
	{
		for(int j =w-2; j >= 0; j--)
		{
			for(int c = 0; c<channel; c++)
			{
				temp.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c) +
					(temp.at<float>(i,(j+1)*channel+c) - temp.at<float>(i,j*channel+c))*V.at<float>(i,j+1);
			}
		}
	}


	for(int i =0; i < h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				output.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c);

	temp.release();
	V.release();


}

void Domain_Filter::compute_boxfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{
	int h = output.rows;
	int w = output.cols;
	Mat lower_pos = Mat(h,w,CV_32FC1);
	Mat upper_pos = Mat(h,w,CV_32FC1);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			lower_pos.at<float>(i,j) = hz.at<float>(i,j) - radius;
			upper_pos.at<float>(i,j) = hz.at<float>(i,j) + radius;
		}

	lower_idx = Mat::zeros(h,w,CV_32FC1);
	upper_idx = Mat::zeros(h,w,CV_32FC1);

	Mat domain_row = Mat::zeros(1,w+1,CV_32FC1);

	for(int i=0;i<h;i++)
	{
		for(int j=0;j<w;j++)
			domain_row.at<float>(0,j) = hz.at<float>(i,j);
		domain_row.at<float>(0,w) = myinf;

		Mat lower_pos_row = Mat::zeros(1,w,CV_32FC1);
		Mat upper_pos_row = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			lower_pos_row.at<float>(0,j) = lower_pos.at<float>(i,j);
			upper_pos_row.at<float>(0,j) = upper_pos.at<float>(i,j);
		}

		Mat temp_lower_idx = Mat::zeros(1,w,CV_32FC1);
		Mat temp_upper_idx = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > lower_pos_row.at<float>(0,0))
			{
				temp_lower_idx.at<float>(0,0) = j;
				break;
			}
		}
		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > upper_pos_row.at<float>(0,0))
			{
				temp_upper_idx.at<float>(0,0) = j;
				break;
			}
		}

		int temp = 0;
		for(int j=1;j<w;j++)
		{
			int count=0;
			for(int k=temp_lower_idx.at<float>(0,j-1);k<w+1;k++)
			{
				if(domain_row.at<float>(0,k) > lower_pos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_lower_idx.at<float>(0,j) = temp_lower_idx.at<float>(0,j-1) + temp;

			count = 0;
			for(int k=temp_upper_idx.at<float>(0,j-1);k<w+1;k++)
			{


				if(domain_row.at<float>(0,k) > upper_pos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_upper_idx.at<float>(0,j) = temp_upper_idx.at<float>(0,j-1) + temp;
		}

		for(int j=0;j<w;j++)
		{
			lower_idx.at<float>(i,j) = temp_lower_idx.at<float>(0,j) + 1;
			upper_idx.at<float>(i,j) = temp_upper_idx.at<float>(0,j) + 1;
		}


		lower_pos_row.release();
		upper_pos_row.release();
		temp_lower_idx.release();
		temp_upper_idx.release();
	}
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			psketch.at<float>(i,j) = upper_idx.at<float>(i,j) - lower_idx.at<float>(i,j);

}
void Domain_Filter::compute_NCfilter(Mat &output, Mat &hz, Mat &psketch, float radius)
{

	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	compute_boxfilter(output,hz,psketch,radius);

	Mat box_filter = Mat::zeros(h,w+1,CV_32FC3);

	for(int i = 0; i < h; i++)
	{
		box_filter.at<float>(i,1*channel+0) = output.at<float>(i,0*channel+0);
		box_filter.at<float>(i,1*channel+1) = output.at<float>(i,0*channel+1);
		box_filter.at<float>(i,1*channel+2) = output.at<float>(i,0*channel+2);
		for(int j = 2; j < w+1; j++)
		{
			for(int c=0;c<channel;c++)
				box_filter.at<float>(i,j*channel+c) = output.at<float>(i,(j-1)*channel+c) + box_filter.at<float>(i,(j-1)*channel+c);
		}
	}

	Mat indices = Mat::zeros(h,w,CV_32FC1);
	Mat final =   Mat::zeros(h,w,CV_32FC3);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			indices.at<float>(i,j) = i+1;

	Mat a = Mat::zeros(h,w,CV_32FC1);
	Mat b = Mat::zeros(h,w,CV_32FC1);

	for(int c=0;c<channel;c++)
	{
		Mat flag = Mat::ones(h,w,CV_32FC1);
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
				flag.at<float>(i,j) = (c+1)*flag.at<float>(i,j);

		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
			{
				a.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (lower_idx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);
				b.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (upper_idx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);

			}

		int p,q,r,rem;
		int p1,q1,r1,rem1;

		for(int i=0;i<h;i++)
		{
			for(int j=0;j<w;j++)
			{

				r = b.at<float>(i,j)/(h*(w+1));
				rem = b.at<float>(i,j) - r*h*(w+1);
				q = rem/h;
				p = rem - q*h;
				if(q==0)
				{
					p=h;
					q=w;
					r=r-1;
				}
				if(p==0)
				{
					p=h;
					q=q-1;
				}
						

				r1 = a.at<float>(i,j)/(h*(w+1));
				rem1 = a.at<float>(i,j) - r1*h*(w+1);
				q1 = rem1/h;
				p1 = rem1 - q1*h;
				if(p1==0)
				{
					p1=h;
					q1=q1-1;
				}


				final.at<float>(i,j*channel+2-c) = (box_filter.at<float>(p-1,q*channel+(2-r)) - box_filter.at<float>(p1-1,q1*channel+(2-r1)))
					/(upper_idx.at<float>(i,j) - lower_idx.at<float>(i,j));
			}
		}
	}

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				output.at<float>(i,j*channel+c) = final.at<float>(i,j*channel+c);


}
void Domain_Filter::init(const Mat &img, int flags, float sigma_s, float sigma_r)
{
	int h = img.size().height;
	int w = img.size().width;
	int channel = img.channels();

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////

	Mat derivx = Mat::zeros(h,w-1,CV_32FC3);
	Mat derivy = Mat::zeros(h-1,w,CV_32FC3);

	diffx(img,derivx);
	diffy(img,derivy);

	Mat distx = Mat::zeros(h,w,CV_32FC1);
	Mat disty = Mat::zeros(h,w,CV_32FC1);

	//////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////

	for(int i = 0; i < h; i++)
		for(int j = 0,k=1; j < w-1; j++,k++)
			for(int c = 0; c < channel; c++)
			{
				distx.at<float>(i,k) = 
					distx.at<float>(i,k) + abs(derivx.at<float>(i,j*channel+c));
			}

	for(int i = 0,k=1; i < h-1; i++,k++)
		for(int j = 0; j < w; j++)
			for(int c = 0; c < channel; c++)
			{
				disty.at<float>(k,j) = 
					disty.at<float>(k,j) + abs(derivy.at<float>(i,j*channel+c));
			}

	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	horiz = Mat(h,w,CV_32FC1);
	vert = Mat(h,w,CV_32FC1);

	Mat final = Mat(h,w,CV_32FC3);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			horiz.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * distx.at<float>(i,j);
			vert.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * disty.at<float>(i,j);
		}


	O = Mat(h,w,CV_32FC3);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				O.at<float>(i,j*channel+c) =  img.at<float>(i,j*channel+c);

	O_t = Mat(w,h,CV_32FC3);

	if(flags == 2)
	{

		ct_H = Mat(h,w,CV_32FC1);
		ct_V = Mat(h,w,CV_32FC1);

		for(int i = 0; i < h; i++)
		{
			ct_H.at<float>(i,0) = horiz.at<float>(i,0);
			for(int j = 1; j < w; j++)
			{
				ct_H.at<float>(i,j) = horiz.at<float>(i,j) + ct_H.at<float>(i,j-1);
			}
		}

		for(int j = 0; j < w; j++)
		{
			ct_V.at<float>(0,j) = vert.at<float>(0,j);
			for(int i = 1; i < h; i++)
			{
				ct_V.at<float>(i,j) = vert.at<float>(i,j) + ct_V.at<float>(i-1,j);
			}
		}
	}

}
void Domain_Filter::filter(const Mat &img, Mat &res, float sigma_s = 60, float sigma_r = 0.4, int flags = 1)
{
	int no_of_iter = 3;
	int h = img.size().height;
	int w = img.size().width;
	float sigma_h = sigma_s;

	init(img,flags,sigma_s,sigma_r);

	if(flags == 1)
	{

		Mat vert_t = vert.t();  

		for(int i=0;i<no_of_iter;i++)
		{
			sigma_h = sigma_s * sqrt(3) * pow(2.0,(no_of_iter - (i+1))) / sqrt(pow(4.0,no_of_iter) -1);

			compute_Rfilter(O, horiz, sigma_h);

			O_t = O.t();

			compute_Rfilter(O_t, vert_t, sigma_h);

			O = O_t.t();

		}
	}
	else if(flags == 2)
	{

		Mat vert_t = ct_V.t();
		Mat temp = Mat(h,w,CV_32FC1);
		Mat temp1 = Mat(w,h,CV_32FC1);

		float radius;

		for(int i=0;i<no_of_iter;i++)
		{
			sigma_h = sigma_s * sqrt(3) * pow(2.0,(no_of_iter - (i+1))) / sqrt(pow(4.0,no_of_iter) -1);

			radius = sqrt(3) * sigma_h;

			compute_NCfilter(O, ct_H, temp,radius);

			O_t = O.t();

			compute_NCfilter(O_t, vert_t, temp1, radius);

			O = O_t.t();
		}
	}

	res = O.clone();
}
