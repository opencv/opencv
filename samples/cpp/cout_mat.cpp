/*
 *
 * cvout_sample just demonstrates the serial out capabilities of cv::Mat
 *  That is, cv::Mat M(...); cout << M;  Now works.
 *
 */

#include "opencv2/core/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void help()
{
	cout
	<< "\n------------------------------------------------------------------\n"
	<< " This program shows the serial out capabilities of cv::Mat\n"
	<< "That is, cv::Mat M(...); cout << M;  Now works.\n"
	<< "Output can be formated to OpenCV, python, numpy, csv and C styles"
	<< "Usage:\n"
	<< "./cvout_sample\n"
	<< "------------------------------------------------------------------\n\n"
	<< endl;
}


int main(int,char**)
{
	help();
    Mat i = Mat::eye(4, 4, CV_64F);
    i.at<double>(1,1) = CV_PI;
    cout << "i = " << i << ";" << endl;

    Mat r = Mat(10, 3, CV_8UC3);
    randu(r, Scalar::all(0), Scalar::all(255));

    cout << "r (default) = " << r << ";" << endl << endl;
    cout << "r (python) = " << format(r,"python") << ";" << endl << endl;
    cout << "r (numpy) = " << format(r,"numpy") << ";" << endl << endl;
    cout << "r (csv) = " << format(r,"csv") << ";" << endl << endl; 
    cout << "r (c) = " << format(r,"C") << ";" << endl << endl;

    Point2f p(5, 1);
    cout << "p = " << p << ";" << endl;

    Point3f p3f(2, 6, 7);
    cout << "p3f = " << p3f << ";" << endl;

    vector<float> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    
    cout << "shortvec = " << Mat(v) << endl;
        
    vector<Point2f> points(20);
    for (size_t i = 0; i < points.size(); ++i)
        points[i] = Point2f((float)(i * 5), (float)(i % 7));

    cout << "points = " << points << ";" << endl;
    return 0;
}
