#include <iostream>
#include "opencv2/photo.hpp"
#include "color_constancy.hpp"

using namespace std;
using namespace cv;

void cv::greyEdge(InputArray _src, OutputArray _dst, int diff_order, int mink_norm, int sigma)
{
    Mat source = _src.getMat();
    _dst.create(source.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    Mat img = Mat(source.size(),CV_32FC3);
    source.convertTo(img,CV_32FC3,1.0);

    double white_R,white_G,white_B;
    Constancy obj;

    Mat output = Mat(img.size(),CV_32FC3);

    obj.general_cc(img,diff_order,mink_norm,sigma,white_R,white_G,white_B,output);

    output.convertTo(dst,CV_8UC3,1);
}

void cv::weightedGreyEdge(InputArray _src, OutputArray _dst, int kappa, int mink_norm, int sigma)
{
    Mat source = _src.getMat();
    _dst.create(source.size(), CV_8UC3);
    Mat dst = _dst.getMat();

    Mat img = Mat(source.size(),CV_32FC3);
    source.convertTo(img,CV_32FC3,1.0);

    double white_R,white_G,white_B;
    Constancy obj;

    Mat output = Mat(img.size(),CV_32FC3);

    obj.weightedGE(img,kappa,mink_norm,sigma,white_R,white_G,white_B,output);

    output.convertTo(dst,CV_8UC3,1);
}
