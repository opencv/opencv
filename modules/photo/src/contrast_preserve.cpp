#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include "math.h"
#include <vector>
#include <limits>
#include <iostream>
#include "contrast_preserve.hpp"

using namespace std;
using namespace cv;

double norm(double);

double norm(double E)
{
        return (sqrt(pow(E,2)));
}

void cv::decolor(InputArray _src, OutputArray _dst, OutputArray _boost)
{
    Mat I = _src.getMat();
    _dst.create(I.size(), CV_8UC1);
    Mat dst = _dst.getMat();

    _boost.create(I.size(), CV_8UC3);
    Mat color_boost = _boost.getMat();

    if(!I.data )
    {
		cout <<  "Could not open or find the image" << endl ;
		return;
	}
	if(I.channels() !=3)
	{
		cout << "Input Color Image" << endl;
		return;
	}

	int maxIter = 15;
	int iterCount = 0;
    float tol = .0001;
    double E = 0;
    double pre_E = std::numeric_limits<double>::infinity();

	Decolor obj;

	Mat img;

    img = Mat(I.size(),CV_32FC3);
    I.convertTo(img,CV_32FC3,1.0/255.0);

    obj.init();

	vector <double> Cg;
	vector < vector <double> > polyGrad;
	vector < vector <double> > bc;
	vector < vector < int > > comb;

	vector <double> alf;

	obj.grad_system(img,polyGrad,Cg,comb);
	obj.weak_order(img,alf);

	Mat Mt = Mat(polyGrad.size(),polyGrad[0].size(), CV_32FC1);
	obj.wei_update_matrix(polyGrad,Cg,Mt);

	vector <double> wei;
	obj.wei_inti(comb,wei);

	//////////////////////////////// main loop starting ////////////////////////////////////////

	while(norm(E-pre_E) > tol)
	{
		iterCount +=1;
        pre_E = E;

		vector <double> G_pos;
		vector <double> G_neg;

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

		double ans = 0.0;
		double ans1 = 0.0;
		for(unsigned int i =0;i<alf.size();i++)
		{
			ans = ((1 + alf[i])/2) * exp((-1.0 * 0.5 * pow(temp[i],2))/pow(sigma,2));
			ans1 =((1 - alf[i])/2) * exp((-1.0 * 0.5 * pow(temp1[i],2))/pow(sigma,2));
			G_pos.push_back(ans);
			G_neg.push_back(ans1);
		}

		vector <double> EXPsum;
		vector <double> EXPterm;

		for(unsigned int i = 0;i<G_pos.size();i++)
			EXPsum.push_back(G_pos[i]+G_neg[i]);


		vector <double> temp2;

		for(unsigned int i=0;i<EXPsum.size();i++)
		{
			if(EXPsum[i] == 0)
				temp2.push_back(1.0);
			else
				temp2.push_back(0.0);
		}

		for(unsigned int i =0; i < G_pos.size();i++)
			EXPterm.push_back((G_pos[i] - G_neg[i])/(EXPsum[i] + temp2[i]));

		
		double val1 = 0.0;
		vector <double> wei1;

		for(unsigned int i=0;i< polyGrad.size();i++)
		{
			val1 = 0.0;
			for(unsigned int j =0;j<polyGrad[0].size();j++)
			{
				val1 = val1 + (Mt.at<float>(i,j) * EXPterm[j]);
			}
			wei1.push_back(val1);
		}

		for(unsigned int i =0;i<wei.size();i++)
			wei[i] = wei1[i];

        E = obj.energyCalcu(Cg,polyGrad,wei);

        if(iterCount > maxIter)
            break;

		G_pos.clear();
		G_neg.clear();
		temp.clear();
		temp1.clear();
		EXPsum.clear();
		EXPterm.clear();
		temp2.clear();
		wei1.clear();
	}

	Mat Gray = Mat::zeros(img.size(),CV_32FC1);
	obj.grayImContruct(wei, img, Gray);

	Gray.convertTo(dst,CV_8UC1,255);

    ///////////////////////////////////       Contrast Boosting   /////////////////////////////////
	
	Mat lab = Mat(img.size(),CV_8UC3);
	Mat color = Mat(img.size(),CV_8UC3);
	Mat l = Mat(img.size(),CV_8UC1);
	Mat a = Mat(img.size(),CV_8UC1);
	Mat b = Mat(img.size(),CV_8UC1);

	cvtColor(I,lab,COLOR_BGR2Lab);

	int h1 = img.size().height;
	int w1 = img.size().width;

	for(int i =0;i<h1;i++)
		for(int j=0;j<w1;j++)
		{
			l.at<uchar>(i,j) = lab.at<uchar>(i,j*3+0);
			a.at<uchar>(i,j) = lab.at<uchar>(i,j*3+1);
			b.at<uchar>(i,j) = lab.at<uchar>(i,j*3+2);
		}
	
	for(int i =0;i<h1;i++)
		for(int j=0;j<w1;j++)
		{
			l.at<uchar>(i,j) = 255.0*Gray.at<float>(i,j);
		}

	for(int i =0;i<h1;i++)
		for(int j=0;j<w1;j++)
		{
			lab.at<uchar>(i,j*3+0) = l.at<uchar>(i,j);
			lab.at<uchar>(i,j*3+1) = a.at<uchar>(i,j);
			lab.at<uchar>(i,j*3+2) = b.at<uchar>(i,j);
		}

	cvtColor(lab,color_boost,COLOR_Lab2BGR);
}
