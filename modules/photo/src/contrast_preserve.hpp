#include "precomp.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
#include <vector>
#include <limits>

using namespace std;
using namespace cv;

class Decolor
{
        public:
                Mat kernel;
                Mat kernel1;
                int order;
                void init();
                void grad_system(Mat img, vector < vector < double > > &polyGrad, vector < double > &Cg, vector < vector <int> >& comb);
};

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
