#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
#include <vector>
#include <limits>
#include <iostream>

#include "contrast_preserve.hpp"

using namespace std;
using namespace cv;
int rounding(double);

int rounding(double a)
{
        return int(a + 0.5);
}

void cv::decolor(InputArray _src, OutputArray _gray, OutputArray _boost)
{
        Mat I = _src.getMat();
        _gray.create(I.size(), CV_8UC1);
        Mat dst = _gray.getMat();

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

        float sigma = .02;
        int maxIter = 8;
        int iterCount = 0;

        int h = I.size().height;
        int w = I.size().width;

        Mat img;
        Decolor obj;

        double sizefactor;

        if((h + w) > 900)
        {
                sizefactor = (double)900/(h+w);
                resize(I,I,Size(rounding(h*sizefactor),rounding(w*sizefactor)));
                img = Mat(I.size(),CV_32FC3);
                I.convertTo(img,CV_32FC3,1.0/255.0);
        }
        else
        {
                img = Mat(I.size(),CV_32FC3);
                I.convertTo(img,CV_32FC3,1.0/255.0);
        }

        obj.init();

        vector <double> Cg;
        vector < vector <double> > polyGrad;
        vector < vector <double> > bc;
        vector < vector < int > > comb;

        vector <double> alf;


}

